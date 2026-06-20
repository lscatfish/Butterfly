"""检查 $$ 公式块"""
import re
from pathlib import Path

md = Path('temp_combined.md').read_text(encoding='utf-8')

# Find all $$...$$ blocks
blocks = re.findall(r'\$\$(.+?)\$\$', md, re.DOTALL)
print(f'Total $$ blocks: {len(blocks)}')

for i, block in enumerate(blocks):
    block = block.strip()
    # Check if it's a simple formula that might not convert
    if not block.startswith('\\begin'):
        if len(block) < 200:
            print(f'Block {i+1}: {block[:100]}')
