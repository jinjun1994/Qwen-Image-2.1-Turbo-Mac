"""Localize only interface fields; prompts, file paths and diagnostic logs are data."""
import json
from pathlib import Path
import re

MESSAGES=json.loads((Path(__file__).parent/'assets/web/i18n.json').read_text())
_PATTERN=[]
for english,chinese in MESSAGES.items():
    if re.search(r'\{\w+\}',english):
        keys=re.findall(r'\{(\w+)\}',english)
        expression=re.escape(english)
        for key in keys:expression=expression.replace(re.escape('{'+key+'}'),f'(?P<{key}>.+)')
        _PATTERN.append((re.compile('^'+expression+'$',re.DOTALL),chinese))


def language(header=''):
    """Use the highest-priority supported language, defaulting to English."""
    choices=[]
    for index,part in enumerate(header.split(',')):
        name,*parameters=part.strip().lower().split(';')
        quality=1.0
        for parameter in parameters:
            if parameter.strip().startswith('q='):
                try:quality=float(parameter.strip()[2:])
                except ValueError:quality=0
        if quality<=0:continue
        if name.startswith('zh'):choices.append((quality,-index,'zh'))
        elif name.startswith('en'):choices.append((quality,-index,'en'))
    return max(choices)[2] if choices else 'en'


def localize(text,locale='en'):
    if not isinstance(text,str) or locale!='zh':return text
    if text in MESSAGES:return MESSAGES[text]
    for pattern,template in _PATTERN:
        match=pattern.fullmatch(text)
        if match:return template.format(**match.groupdict())
    return text


def localize_payload(value,locale):
    if isinstance(value,list):return [localize_payload(item,locale) for item in value]
    if not isinstance(value,dict):return value
    return {key:(localize(item,locale) if key in ('error','engine_error','stage') else
                 localize_payload(item,locale) if key not in ('metadata','params','logs','values','models','catalog') else item)
            for key,item in value.items()}
