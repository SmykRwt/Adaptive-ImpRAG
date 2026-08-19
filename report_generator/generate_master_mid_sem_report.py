import os
import sys
from create_full_docx import FullReportGenerator
from build_part1_front_and_intro import add_front_matter_and_chapter1
from build_part2_req_analysis import add_chapter2_requirement_analysis
from build_part3_methodology import add_chapter3_methodology
from build_part4_design_and_proto import add_chapter4_design_and_prototype
from build_part5_conclusions_and_refs import add_chapter5_conclusions_and_refs

def main():
    template_docx = "Adaptive ImpRAG report final (2).docx"
    output_docx = "Adaptive_ImpRAG_Mid_Semester_Report_2026.docx"
    
    print(f"Starting master report generation...")
    print(f"Template base: {template_docx}")
    print(f"Output target: {output_docx}")
    
    gen = FullReportGenerator(template_docx, output_docx)
    
    print("Building Part 1: Front Matter, Title Page, Abstract, TOC, and Chapter 1 (Introduction)...")
    add_front_matter_and_chapter1(gen)
    
    print("Building Part 2: Chapter 2 (Requirement Analysis, Literature Survey, SRS, Cost & Risk Analysis)...")
    add_chapter2_requirement_analysis(gen)
    
    print("Building Part 3: Chapter 3 (Methodology Adopted, 8 Subsystems, WBS, Technology Stack)...")
    add_chapter3_methodology(gen)
    
    print("Building Part 4: Chapter 4 (Design Specifications, 7 Mermaid Diagrams, Prototype Walkthrough)...")
    add_chapter4_design_and_prototype(gen)
    
    print("Building Part 5: Chapter 5 (Conclusions, Benefits, Roadmap, References, Plagiarism Summary)...")
    add_chapter5_conclusions_and_refs(gen)
    
    print("Packing and saving final .docx file...")
    gen.save()
    print("Master report generation complete!")

if __name__ == "__main__":
    main()
