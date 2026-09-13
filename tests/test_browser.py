"""Opt-in end-to-end browser check. Run with TISSUE_BROWSER_TEST=1 pytest -m browser."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import pytest


@pytest.mark.browser
@pytest.mark.skipif(os.environ.get('TISSUE_BROWSER_TEST') != '1',reason='Opt-in browser integration test')
@pytest.mark.parametrize('geometry', ['square', 'sphere'])
def test_dashboard_controls(tmp_path, geometry):
    from playwright.sync_api import sync_playwright, expect
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    config=Path(__file__).resolve().parents[1]/'configs'/('sphere.json' if geometry == 'sphere' else 'quick_demo.json')
    backend=os.environ.get('TISSUE_TEST_BACKEND','scipy')
    log=(tmp_path/'server.log').open('w')
    server=subprocess.Popen([sys.executable,'-m','tissue_growth','dashboard','--config',str(config),'--geometry',geometry,'--backend',backend,'--n','16','--dt','0.001','--t-end','1000','--port',str(port),'--output',str(tmp_path/'runs')],stdout=log,stderr=subprocess.STDOUT)
    try:
        url=f'http://127.0.0.1:{port}'
        for _ in range(300):
            assert server.poll() is None,(tmp_path/'server.log').read_text()
            try:
                urllib.request.urlopen(url,timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError((tmp_path/'server.log').read_text())
        with sync_playwright() as p:
            executable=os.environ.get('TISSUE_CHROMIUM')
            browser=p.chromium.launch(headless=True,executable_path=executable,args=['--use-gl=angle','--use-angle=swiftshader'])
            page=browser.new_page(viewport={'width':1440,'height':1000})
            errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(url)
            expect(page.get_by_role('button',name='Run',exact=True)).to_be_enabled(timeout=60000)
            expect(page.locator('canvas').first).to_be_visible()
            if geometry == 'sphere':
                expect(page.get_by_label('Longitude / 360°')).to_be_visible()
                expect(page.locator('.metric-label').nth(1)).to_have_text('Radius')
            page.screenshot(path=os.environ.get('TISSUE_SCREENSHOT',str(tmp_path/'initial.png')))
            page.get_by_role('button',name='Apply perturbation',exact=True).click()
            expect(page.locator('#run-status')).to_contain_text('Perturb applied',timeout=15000)
            page.get_by_role('button',name='Run',exact=True).click()
            expect(page.get_by_role('button',name='Pause',exact=True)).to_be_enabled()
            page.wait_for_timeout(1000)
            page.get_by_role('button',name='Pause',exact=True).click()
            expect(page.get_by_role('button',name='Run',exact=True)).to_be_enabled()
            time_before=page.locator('.metric-value').nth(0).inner_text()
            page.wait_for_timeout(600)
            assert page.locator('.metric-value').nth(0).inner_text()==time_before
            assert float(time_before)>0
            page.get_by_label('Linear growth rate (1 / time)').fill('0.03')
            page.get_by_role('button',name='Apply growth rate',exact=True).click()
            expect(page.locator('#run-status')).to_contain_text('Growth applied')
            page.get_by_role('button',name='Save checkpoint',exact=True).click()
            expect(page.locator('#run-status')).to_contain_text('Save applied')
            assert list((tmp_path/'runs').glob('*/checkpoint.npz'))
            screenshot=os.environ.get('TISSUE_SCREENSHOT')
            if screenshot:
                page.screenshot(path=screenshot)
            assert not errors,errors
            assert page.locator('.error').count()==0,page.locator('.error').all_text_contents()
            page.get_by_role('button',name='Reset',exact=True).click()
            expect(page.get_by_role('button',name='Run',exact=True)).to_be_enabled(timeout=60000)
            expect(page.locator('.metric-value').nth(0)).to_have_text('0.00')
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        log.close()
