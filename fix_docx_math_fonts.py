"""
统一 DOCX 中所有数学公式（m:oMath）的字体为 Cambria Math。

用法：
    python fix_docx_math_fonts.py [input.docx] [output.docx]

默认：
    input  = output/仿生蝴蝶扑翼MAV机械原理分析报告.docx
    output = output/仿生蝴蝶扑翼MAV机械原理分析报告_math_fixed.docx

说明：
- 仅修改 word/document.xml 中 m:oMath 下的 m:r 字体。
- 保留现有 OMML 结构（分数、上下标、根号等）。
- 不修改 convert_to_docx.py，可独立使用。
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# 注册命名空间，保证写出时前缀与 Word 一致
ET.register_namespace("m", MATH_NS)
ET.register_namespace("w", WORD_NS)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
ET.register_namespace("pic", "http://schemas.openxmlformats.org/drawingml/2006/picture")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")
ET.register_namespace("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")


def set_cambria_math_fonts(run_elem):
    """确保 m:r 元素内的 w:rPr/w:rFonts 使用 Cambria Math。"""
    # 找到或创建 m:rPr（保留原有 m:sty 等属性）
    rpr = run_elem.find(f"{{{WORD_NS}}}rPr")
    if rpr is None:
        rpr = ET.Element(f"{{{WORD_NS}}}rPr")
        run_elem.insert(0, rpr)

    # 找到或创建 w:rFonts
    rfonts = rpr.find(f"{{{WORD_NS}}}rFonts")
    if rfonts is None:
        rfonts = ET.SubElement(rpr, f"{{{WORD_NS}}}rFonts")

    # 设置字体属性
    rfonts.set(f"{{{WORD_NS}}}ascii", "Cambria Math")
    rfonts.set(f"{{{WORD_NS}}}hAnsi", "Cambria Math")
    rfonts.set(f"{{{WORD_NS}}}cs", "Cambria Math")
    # 与 Word 默认行为一致，不强制 eastAsia，避免中文标点走形
    # 可选：Word 内部 hint="default"，加了更贴近原始调好的公式
    if not rfonts.get(f"{{{WORD_NS}}}hint"):
        rfonts.set(f"{{{WORD_NS}}}hint", "default")


def fix_document_xml(xml_text: str) -> str:
    root = ET.fromstring(xml_text.encode("utf-8"))

    # 定位所有 m:oMath（含嵌套）
    for omath in root.iter(f"{{{MATH_NS}}}oMath"):
        for run in omath.iter(f"{{{MATH_NS}}}r"):
            set_cambria_math_fonts(run)

    # 写回字符串，保留 XML 声明
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(
        root, encoding="unicode"
    )


def fix_docx(input_path: Path, output_path: Path):
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(input_path, "r") as zin, zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = fix_document_xml(data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)

    print(f"已生成：{output_path}")


def main():
    default_input = Path(__file__).parent / "output" / "仿生蝴蝶扑翼MAV机械原理分析报告.docx"
    parser = argparse.ArgumentParser(description="统一 DOCX 公式字体为 Cambria Math")
    parser.add_argument("input", nargs="?", type=Path, default=default_input)
    parser.add_argument("output", nargs="?", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.input.resolve()
    if args.output is None:
        output_path = input_path.with_stem(input_path.stem + "_math_fixed")
    else:
        output_path = args.output.resolve()

    if not input_path.exists():
        print(f"错误：找不到输入文件 {input_path}", file=sys.stderr)
        sys.exit(1)

    fix_docx(input_path, output_path)


if __name__ == "__main__":
    main()
