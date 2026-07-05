import xml.etree.ElementTree as ET

tree = ET.parse('unpacked_report/word/document.xml')
root = tree.getroot()

# Search for any paragraph containing 齿轮参数
found = False
for i, p in enumerate(root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')):
    texts = []
    for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t.text:
            texts.append(t.text)
    full = ''.join(texts)
    
    if '齿轮参数' in full:
        print(f'=== 段落 {i} ===')
        print(f'文本：{full[:150]}')
        
        run_count = 0
        for r in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r'):
            run_texts = []
            for t in r.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                if t.text:
                    run_texts.append(t.text)
            run_text = ''.join(run_texts)
            
            has_math = any(child.tag.endswith('oMath') for child in r)
            print(f'  Run {run_count}: "{run_text[:60]}" math={has_math}')
            run_count += 1
        
        found = True
        if run_count > 5:
            break

if not found:
    print('未找到包含"齿轮参数"的段落')
