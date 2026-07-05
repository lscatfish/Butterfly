from docx import Document
from docx.oxml.ns import qn

doc = Document('output/仿生蝴蝶扑翼MAV机械原理分析报告.docx')

block_count = 0
inline_count = 0
block_total = 0
inline_total = 0

for para in doc.paragraphs:
    # Block-level formulas (direct children of paragraph)
    for math in para._element.findall(qn('m:oMath')):
        block_total += 1
        rpr = math.find(qn('m:rPr'))
        if rpr is not None:
            rFonts = rpr.find(qn('m:rFonts'))
            if rFonts is not None and rFonts.get(qn('m:ascii')) == 'Cambria Math':
                block_count += 1

    # Inline formulas (nested inside runs)
    for run in para.runs:
        for math in run._element.findall(qn('m:oMath')):
            inline_total += 1
            rpr = math.find(qn('m:rPr'))
            if rpr is not None:
                rFonts = rpr.find(qn('m:rFonts'))
                if rFonts is not None and rFonts.get(qn('m:ascii')) == 'Cambria Math':
                    inline_count += 1

print(f'块公式: {block_count}/{block_total}')
print(f'行内公式: {inline_count}/{inline_total}')
print(f'总计: {block_count + inline_count}/{block_total + inline_total}')
