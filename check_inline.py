from docx import Document
from docx.oxml.ns import qn

doc = Document('output/仿生蝴蝶扑翼MAV机械原理分析报告.docx')

# Find a paragraph with inline math
for para in doc.paragraphs:
    if '模数' in para.text and 'mm' in para.text:
        print(f'段落文本：{para.text[:100]}')
        print(f'Run 数量：{len(para.runs)}')
        for i, run in enumerate(para.runs):
            text = run.text if run.text else '(empty)'
            print(f'  Run {i}: {text[:50]}')
            # Check for math elements in run
            math_elems = run._element.findall(qn('m:oMath'))
            if math_elems:
                print(f'    -> 找到 {len(math_elems)} 个 m:oMath')
        break
