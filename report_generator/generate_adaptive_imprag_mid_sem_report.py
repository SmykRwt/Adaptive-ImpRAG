import zipfile
import os
import xml.sax.saxutils as saxutils

def escape(t):
    return saxutils.escape(str(t))

class FullReportBuilder:
    def __init__(self, template_docx_path, output_docx_path):
        self.template_docx_path = template_docx_path
        self.output_docx_path = output_docx_path
        self.body_elements = []
        
    def add_page_break(self):
        self.body_elements.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def add_title_p(self, text, size=36, bold=True, color="1F4E79", align="center", space_before=200, space_after=200):
        jc = f'<w:jc w:val="{align}"/>'
        sp = f'<w:spacing w:before="{space_before}" w:after="{space_after}"/>'
        rpr = f'<w:rPr><w:b/><w:bCs/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:color w:val="{color}"/></w:rPr>'
        self.body_elements.append(f'''<w:p><w:pPr>{jc}{sp}</w:pPr><w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>''')

    def add_heading_1(self, text):
        # Chapter Heading
        p = f'''<w:p>
            <w:pPr>
                <w:pStyle w:val="Heading1"/>
                <w:spacing w:before="360" w:after="160"/>
                <w:rPr>
                    <w:b/><w:bCs/>
                    <w:sz w:val="32"/>
                    <w:szCs w:val="32"/>
                    <w:color w:val="1F4E79"/>
                </w:rPr>
            </w:pPr>
            <w:r>
                <w:rPr>
                    <w:b/><w:bCs/>
                    <w:sz w:val="32"/>
                    <w:szCs w:val="32"/>
                    <w:color w:val="1F4E79"/>
                </w:rPr>
                <w:t xml:space="preserve">{escape(text)}</w:t>
            </w:r>
        </w:p>'''
        self.body_elements.append(p)

    def add_heading_2(self, text):
        p = f'''<w:p>
            <w:pPr>
                <w:pStyle w:val="Heading2"/>
                <w:spacing w:before="240" w:after="100"/>
                <w:rPr>
                    <w:b/><w:bCs/>
                    <w:sz w:val="28"/>
                    <w:szCs w:val="28"/>
                    <w:color w:val="2E75B6"/>
                </w:rPr>
            </w:pPr>
            <w:r>
                <w:rPr>
                    <w:b/><w:bCs/>
                    <w:sz w:val="28"/>
                    <w:szCs w:val="28"/>
                    <w:color w:val="2E75B6"/>
                </w:rPr>
                <w:t xml:space="preserve">{escape(text)}</w:t>
            </w:r>
        </w:p>'''
        self.body_elements.append(p)

    def add_heading_3(self, text):
        p = f'''<w:p>
            <w:pPr>
                <w:pStyle w:val="Heading3"/>
                <w:spacing w:before="160" w:after="80"/>
                <w:rPr>
                    <w:b/><w:bCs/>
                    <w:sz w:val="24"/>
                    <w:szCs w:val="24"/>
                    <w:color w:val="1F4E79"/>
                </w:rPr>
            </w:pPr>
            <w:r>
                <w:rPr>
                    <w:b/><w:bCs/>
                    <w:sz w:val="24"/>
                    <w:szCs w:val="24"/>
                    <w:color w:val="1F4E79"/>
                </w:rPr>
                <w:t xml:space="preserve">{escape(text)}</w:t>
            </w:r>
        </w:p>'''
        self.body_elements.append(p)

    def add_heading_4(self, text):
        p = f'''<w:p>
            <w:pPr>
                <w:pStyle w:val="Heading4"/>
                <w:spacing w:before="120" w:after="60"/>
                <w:rPr>
                    <w:b/><w:bCs/><w:i/><w:iCs/>
                    <w:sz w:val="24"/>
                    <w:szCs w:val="24"/>
                    <w:color w:val="2E75B6"/>
                </w:rPr>
            </w:pPr>
            <w:r>
                <w:rPr>
                    <w:b/><w:bCs/><w:i/><w:iCs/>
                    <w:sz w:val="24"/>
                    <w:szCs w:val="24"/>
                    <w:color w:val="2E75B6"/>
                </w:rPr>
                <w:t xml:space="preserve">{escape(text)}</w:t>
            </w:r>
        </w:p>'''
        self.body_elements.append(p)

    def add_p(self, text, bold=False, italic=False, align="both", size=24, space_after=120, space_before=0, color="000000"):
        jc = f'<w:jc w:val="{align}"/>'
        sp = f'<w:spacing w:before="{space_before}" w:after="{space_after}" w:line="276" w:lineRule="auto"/>'
        rpr_items = [f'<w:sz w:val="{size}"/>', f'<w:szCs w:val="{size}"/>']
        if bold: rpr_items.append('<w:b/><w:bCs/>')
        if italic: rpr_items.append('<w:i/><w:iCs/>')
        if color and color != "000000": rpr_items.append(f'<w:color w:val="{color}"/>')
        rpr = f'<w:rPr>{"".join(rpr_items)}</w:rPr>'
        
        p = f'''<w:p>
            <w:pPr>
                <w:pStyle w:val="Normal"/>
                {jc}
                {sp}
            </w:pPr>
            <w:r>
                {rpr}
                <w:t xml:space="preserve">{escape(text)}</w:t>
            </w:r>
        </w:p>'''
        self.body_elements.append(p)

    def add_bullet(self, text, level=0, bold_prefix=None):
        indent = 360 * (level + 1)
        sp = '<w:spacing w:before="40" w:after="60" w:line="260" w:lineRule="auto"/>'
        
        runs = []
        if bold_prefix:
            runs.append(f'''<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="24"/><w:szCs w:val="24"/><w:color w:val="1F4E79"/></w:rPr><w:t xml:space="preserve">{escape(bold_prefix)} </w:t></w:r>''')
        runs.append(f'''<w:r><w:rPr><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>''')
        
        p = f'''<w:p>
            <w:pPr>
                <w:pStyle w:val="ListBullet"/>
                <w:ind w:left="{indent}" w:hanging="240"/>
                {sp}
            </w:pPr>
            <w:r><w:rPr><w:sz w:val="24"/><w:color w:val="2E75B6"/></w:rPr><w:t>• </w:t></w:r>
            {"".join(runs)}
        </w:p>'''
        self.body_elements.append(p)

    def add_callout(self, text, title="IMPORTANT ARCHITECTURAL NOTE"):
        p = f'''<w:p>
            <w:pPr>
                <w:pBdr>
                    <w:left w:val="single" w:sz="36" w:space="15" w:color="2E75B6"/>
                </w:pBdr>
                <w:shd w:val="clear" w:color="auto" w:fill="F0F4F8"/>
                <w:spacing w:before="120" w:after="120"/>
                <w:ind w:left="240" w:right="120"/>
            </w:pPr>
            <w:r>
                <w:rPr><w:b/><w:bCs/><w:color w:val="1F4E79"/><w:sz w:val="22"/></w:rPr>
                <w:t xml:space="preserve">[{escape(title)}]: </w:t>
            </w:r>
            <w:r>
                <w:rPr><w:i/><w:iCs/><w:sz w:val="22"/></w:rPr>
                <w:t xml:space="preserve">{escape(text)}</w:t>
            </w:r>
        </w:p>'''
        self.body_elements.append(p)

    def add_table(self, headers, rows, col_widths=None):
        tbl_pr = '''<w:tblPr>
            <w:tblW w:w="5000" w:type="pct"/>
            <w:tblBorders>
                <w:top w:val="single" w:sz="6" w:space="0" w:color="B0C4DE"/>
                <w:left w:val="single" w:sz="6" w:space="0" w:color="B0C4DE"/>
                <w:bottom w:val="single" w:sz="8" w:space="0" w:color="2E75B6"/>
                <w:right w:val="single" w:sz="6" w:space="0" w:color="B0C4DE"/>
                <w:insideH w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>
                <w:insideV w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>
            </w:tblBorders>
            <w:tblCellMar>
                <w:top w:w="120" w:type="dxa"/>
                <w:left w:w="160" w:type="dxa"/>
                <w:bottom w:w="120" w:type="dxa"/>
                <w:right w:w="160" w:type="dxa"/>
            </w:tblCellMar>
        </w:tblPr>'''
        
        # Header Row
        header_cells = []
        for i, h in enumerate(headers):
            w_str = f'<w:tcW w:w="{col_widths[i]}" w:type="dxa"/>' if col_widths else '<w:tcW w:w="0" w:type="auto"/>'
            cell = f'''<w:tc>
                <w:tcPr>
                    {w_str}
                    <w:shd w:val="clear" w:color="auto" w:fill="2E75B6"/>
                </w:tcPr>
                <w:p>
                    <w:pPr><w:jc w:val="center"/><w:spacing w:before="60" w:after="60"/></w:pPr>
                    <w:r>
                        <w:rPr><w:b/><w:bCs/><w:color w:val="FFFFFF"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
                        <w:t xml:space="preserve">{escape(h)}</w:t>
                    </w:r>
                </w:p>
            </w:tc>'''
            header_cells.append(cell)
            
        header_tr = f'''<w:tr>
            <w:trPr><w:tblHeader/></w:trPr>
            {"".join(header_cells)}
        </w:tr>'''
        
        # Body Rows
        body_trs = []
        for r_idx, row in enumerate(rows):
            fill_color = "F9FBFC" if r_idx % 2 == 1 else "FFFFFF"
            cells = []
            for c_idx, val in enumerate(row):
                w_str = f'<w:tcW w:w="{col_widths[c_idx]}" w:type="dxa"/>' if col_widths else '<w:tcW w:w="0" w:type="auto"/>'
                align = "center" if c_idx == 0 and len(val) < 6 else "left"
                cell = f'''<w:tc>
                    <w:tcPr>
                        {w_str}
                        <w:shd w:val="clear" w:color="auto" w:fill="{fill_color}"/>
                    </w:tcPr>
                    <w:p>
                        <w:pPr><w:jc w:val="{align}"/><w:spacing w:before="40" w:after="40"/><w:line w:line="240" w:lineRule="auto"/></w:pPr>
                        <w:r>
                            <w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>
                            <w:t xml:space="preserve">{escape(str(val))}</w:t>
                        </w:r>
                    </w:p>
                </w:tc>'''
                cells.append(cell)
            body_trs.append(f'<w:tr>{"".join(cells)}</w:tr>')
            
        tbl_xml = f'''<w:tbl>
            {tbl_pr}
            {header_tr}
            {"".join(body_trs)}
        </w:tbl>'''
        self.body_elements.append(tbl_xml)
        self.body_elements.append('<w:p><w:pPr><w:spacing w:before="60" w:after="120"/></w:pPr></w:p>')

    def add_diagram_box(self, fig_num, caption, mermaid_code, explanation):
        # Frame box with diagram code
        code_lines = mermaid_code.strip().splitlines()
        code_paragraphs = []
        for line in code_lines:
            code_paragraphs.append(f'''<w:p>
                <w:pPr>
                    <w:spacing w:before="10" w:after="10"/>
                    <w:ind w:left="180"/>
                </w:pPr>
                <w:r>
                    <w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/><w:szCs w:val="18"/><w:color w:val="003366"/></w:rPr>
                    <w:t xml:space="preserve">{escape(line)}</w:t>
                </w:r>
            </w:p>''')
            
        diagram_box = f'''<w:tbl>
            <w:tblPr>
                <w:tblW w:w="5000" w:type="pct"/>
                <w:tblBorders>
                    <w:top w:val="single" w:sz="8" w:space="0" w:color="2E75B6"/>
                    <w:left w:val="single" w:sz="8" w:space="0" w:color="2E75B6"/>
                    <w:bottom w:val="single" w:sz="8" w:space="0" w:color="2E75B6"/>
                    <w:right w:val="single" w:sz="8" w:space="0" w:color="2E75B6"/>
                </w:tblBorders>
                <w:tblCellMar>
                    <w:top w:w="120" w:type="dxa"/>
                    <w:left w:w="160" w:type="dxa"/>
                    <w:bottom w:w="120" w:type="dxa"/>
                    <w:right w:w="160" w:type="dxa"/>
                </w:tblCellMar>
            </w:tblPr>
            <w:tr>
                <w:tc>
                    <w:tcPr>
                        <w:shd w:val="clear" w:color="auto" w:fill="F4F6F9"/>
                    </w:tcPr>
                    <w:p>
                        <w:pPr><w:jc w:val="center"/><w:spacing w:before="60" w:after="60"/></w:pPr>
                        <w:r>
                            <w:rPr><w:b/><w:bCs/><w:color w:val="1F4E79"/><w:sz w:val="20"/></w:rPr>
                            <w:t xml:space="preserve">[STRUCTURED ARCHITECTURAL DIAGRAM IN MERMAID SPECIFICATION]</w:t>
                        </w:r>
                    </w:p>
                    {"".join(code_paragraphs)}
                </w:tc>
            </w:tr>
        </w:tbl>'''
        self.body_elements.append(diagram_box)
        
        # Caption
        caption_p = f'''<w:p>
            <w:pPr>
                <w:jc w:val="center"/>
                <w:spacing w:before="80" w:after="120"/>
            </w:pPr>
            <w:r>
                <w:rPr><w:b/><w:bCs/><w:i/><w:iCs/><w:color w:val="1F4E79"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
                <w:t xml:space="preserve">Figure {fig_num}: {escape(caption)}</w:t>
            </w:r>
        </w:p>'''
        self.body_elements.append(caption_p)
        
        # Explanation
        self.add_p(f"Description of Figure {fig_num}: {explanation}", italic=False, size=22, space_after=160)

    def save(self):
        with zipfile.ZipFile(self.template_docx_path, 'r') as zin:
            file_entries = {name: zin.read(name) for name in zin.namelist()}
            
        full_body_xml = "".join(self.body_elements)
        
        document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
            xmlns:v="urn:schemas-microsoft-com:vml"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:w10="urn:schemas-microsoft-com:office:word"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
    <w:body>
        {full_body_xml}
        <w:sectPr>
            <w:pgSz w:w="12240" w:h="15840"/>
            <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
            <w:cols w:space="720"/>
            <w:docGrid w:linePitch="360"/>
        </w:sectPr>
    </w:body>
</w:document>'''
        
        file_entries['word/document.xml'] = document_xml.encode('utf-8')
        
        with zipfile.ZipFile(self.output_docx_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for name, content in file_entries.items():
                zout.writestr(name, content)
                
        print(f"Report document successfully generated at: {self.output_docx_path}")
        print(f"File size: {os.path.getsize(self.output_docx_path):,} bytes")

print("FullReportBuilder class defined successfully.")
