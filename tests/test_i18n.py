import json
from pathlib import Path
import re
import unittest
from html.parser import HTMLParser
from mqwen.i18n import MESSAGES,language,localize,localize_payload

ROOT=Path(__file__).resolve().parents[1]

class TranslationTests(unittest.TestCase):
    def test_default_and_weighted_language(self):
        for header,expected in [('', 'en'),('fr','en'),('zh-CN','zh'),('zh;q=0.4,en;q=0.9','en'),('en;q=0,zh;q=1','zh'),('zh;q=0','en')]:
            with self.subTest(header=header):self.assertEqual(language(header),expected)

    def test_dynamic_status_and_paths(self):
        self.assertEqual(localize('Denoising 3/8','zh'),'去噪 3/8')
        self.assertEqual(localize('Model not found: /tmp/{name}.safetensors','zh'),'模型不存在: /tmp/{name}.safetensors')
        self.assertEqual(localize('Completed','en'),'Completed')
        self.assertEqual(localize('Unknown library diagnostic','zh'),'Unknown library diagnostic')

    def test_only_presentation_fields_change(self):
        data={'stage':'Completed','jobs':[{'stage':'Denoising 4/8','params':{'prompt':'Completed','stage':'Completed'},'logs':['Completed']}], 'error':'Use an absolute path','values':{'stage':'Completed'},'metadata':{'stage':'Completed'}}
        translated=localize_payload(data,'zh')
        self.assertEqual(translated['stage'],'已完成')
        self.assertEqual(translated['error'],'请使用绝对路径')
        self.assertEqual(translated['jobs'][0]['stage'],'去噪 4/8')
        self.assertEqual(translated['jobs'][0]['params'],data['jobs'][0]['params'])
        self.assertEqual(translated['jobs'][0]['logs'],['Completed'])
        self.assertEqual(translated['values'],data['values'])
        self.assertEqual(translated['metadata'],data['metadata'])
        self.assertEqual(data['stage'],'Completed')

    def test_placeholders_are_preserved(self):
        for english,chinese in MESSAGES.items():
            self.assertTrue(chinese,english)
            self.assertEqual(set(re.findall(r'\{\w+\}',english)),set(re.findall(r'\{\w+\}',chinese)),english)

    def test_static_html_is_english_with_complete_dictionary(self):
        case=self
        class Check(HTMLParser):
            def handle_starttag(self,tag,attrs):
                for key,value in attrs:
                    if key.startswith('data-i18n'):case.assertIn(value,MESSAGES)
                    if tag=='html' and key=='lang':case.assertEqual(value,'en')
            def handle_data(self,data):
                if re.search('[\u4e00-\u9fff]',data):case.assertEqual(data.strip(),'中文')
        Check().feed((ROOT/'src/mqwen/assets/web/index.html').read_text())
        js=(ROOT/'src/mqwen/assets/web/app.js').read_text()
        self.assertIsNone(re.search('[\u4e00-\u9fff]',js))
        for key in re.findall(r"\bt\('([^']+)'",js):self.assertIn(key,MESSAGES)

    def test_native_chrome_keys_are_translated(self):
        swift=(ROOT/'desktop/MQwen.swift').read_text()
        for key in re.findall(r'\btr\("([^"\n]+)"',swift):
            if '\\n' not in key:self.assertIn(key,MESSAGES)

class LocalizedHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os,select,subprocess,sys,tempfile
        cls.folder=tempfile.TemporaryDirectory()
        root=Path(cls.folder.name);output=root/'outputs';output.mkdir()
        (output/'sample.png').write_bytes(b'PNG fixture')
        (output/'sample.json').write_text(json.dumps(dict(backend='mlx-metal',prompt='Completed',seed=42,width=512,height=512)))
        cls.process=subprocess.Popen([sys.executable,'-m','mqwen.cli','web','--desktop','--no-auto-start','--port','0','--output-dir',str(output),'--config-dir',str(root/'config')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
        if not select.select([cls.process.stdout],[],[],10)[0]:raise RuntimeError('HTTP server did not start')
        line=cls.process.stdout.readline();match=re.search(r'http://127.0.0.1:\d+',line)
        if not match:raise RuntimeError(line)
        cls.base=match.group()

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate();cls.process.wait(timeout=10);cls.process.stdout.close();cls.folder.cleanup()

    def request(self,path,locale=None,data=None):
        from urllib.request import Request,build_opener,ProxyHandler
        from urllib.error import HTTPError
        headers={} if locale is None else {'Accept-Language':locale}
        if data is not None:headers['Content-Type']='application/json'
        req=Request(self.base+path,data=None if data is None else json.dumps(data).encode(),headers=headers)
        try:response=build_opener(ProxyHandler({})).open(req,timeout=5)
        except HTTPError as exc:response=exc
        with response:return response.status,json.load(response)

    def test_status_changes_locale_without_touching_prompt(self):
        for locale,stage in [(None,'Completed'),('zh-CN','已完成'),('en','Completed')]:
            code,state=self.request('/api/state',locale)
            self.assertEqual(code,200);self.assertEqual(state['jobs'][0]['stage'],stage)
            self.assertEqual(state['jobs'][0]['params']['prompt'],'Completed')
            self.assertEqual(state['engine_state'],'stopped')

    def test_validation_and_static_catalog(self):
        for locale,error in [(None,'Enter a prompt of 1–12000 characters'),('zh','请输入 1–12000 字符的提示词')]:
            code,data=self.request('/api/generate',locale,{})
            self.assertEqual(code,400);self.assertEqual(data['error'],error)
        code,catalog=self.request('/i18n.json')
        self.assertEqual(code,200);self.assertEqual(catalog['Start inference'],'启动推理服务')

    def test_prompt_presets_preserve_original_text_in_both_languages(self):
        expected=json.loads((ROOT/'src/mqwen/assets/web/prompts.json').read_text())
        self.assertEqual(len(expected),6)
        self.assertEqual(len(set(expected)),6)
        self.assertIn('xiāng jiāo',expected[2])
        for locale in ('en','zh'):
            code,presets=self.request('/prompts.json',locale)
            self.assertEqual(code,200)
            self.assertEqual(presets,expected)
