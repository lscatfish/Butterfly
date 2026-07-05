import xml.etree.ElementTree as ET

tree = ET.parse('unpacked_report/word/document.xml')
root = tree.getroot()

# Find paragraphs with inline math
for p in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
    texts = []
    for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t.text:
            texts.append(t.text)
    full = ''.join(texts)
    
    if '模数' in full or '齿轮参数' in full:
        print(f'段落：{full[:80]}')
        
        # Check block math
        has_block = False
        for child in p:
            if child.tag.endswith('oMath'):
                has_block = True
        
        # Check inline math in runs
        run_count = 0
        for r in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r'):
            run_texts = []
            for t in r.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                if t.text:
                    run_texts.append(t.text)
            run_text = ''.join(run_texts)
            
            has_math = any(child.tag.endswith('oMath') for child in r)
            print(f'  Run {run_count}: "{run_text[:40]}" math={has_math}')
            run_count += 1
        
        print(f'  块公式：{has_block}')
        break
