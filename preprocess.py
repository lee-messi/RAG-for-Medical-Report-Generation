import json
import re
from collections import defaultdict

def fix_organ_typos(organ):
    """Fix organ name typos"""
    if not organ:
        return organ
    
    organ_fixes = {
        "Anus": "Colon",
        "stomach": "Stomach", 
        "Cervix": "Uterine cervix",
        "Bladder": "Urinary bladder",
        "Nipple": "Breast",
        "Rectum": "Colon"
    }
    
    return organ_fixes.get(organ, organ)

def fix_procedure_typos(procedure):
    """Fix procedure typos"""
    if not procedure:
        return procedure
    
    procedure_fixes = {
        "olonoscopic mucosal resection": "colonoscopic mucosal resection",
        "  colonoscopic submucosal dissection": "colonoscopic submucosal dissection"
    }
    
    for typo, fix in procedure_fixes.items():
        procedure = re.sub(re.escape(typo), fix, procedure, flags=re.IGNORECASE)
    
    return procedure.strip()

def fix_diagnosis_typos(diagnosis):
    """Fix diagnosis typos"""
    diagnosis = re.sub(r'(?:,\s*\n\s*with|with\s*\n\s+)', ' with \n', diagnosis)
    
    # Handle Extranodal marginal zone variations
    diagnosis = re.sub(r'Extranodal marginal zone.*', 
                      'Extranodal marginal zone B cell lymphoma of MALT type', 
                      diagnosis, flags=re.IGNORECASE | re.DOTALL)
    
    diagnosis_fixes = {
        "Nuclear grade: low": "Nuclear grade: Low",
        "Low grade dysplasia": "low grade dysplasia",
        "LIntermediate": "Intermediate",
        "focal": "Focal", 
        "oherewise": "otherwise",
        "carinoma": "carcinoma",
        " nterstinal": " intestinal",
        " interstinal": " intestinal",
        "gastrtis": "gastritis",
        "serrated serrated": "serrated",
        "otherewise": "otherwise",
        "probably": ""
    }
    
    for typo, fix in diagnosis_fixes.items():
        diagnosis = re.sub(re.escape(typo), fix, diagnosis, flags=re.IGNORECASE)
    
    return diagnosis

def normalize_spacing(text):
    """Normalize spacing in diagnosis text"""
    if not text:
        return text
    
    # Replace multiple spaces with single space (but preserve newline formatting)
    text = re.sub(r'(?<!\n) +', ' ', text)
    # Fix spacing around colons
    text = re.sub(r':([A-Za-z])', r': \1', text)
    # Normalize spacing after newlines - keep 2 spaces before dash
    text = re.sub(r'\n +- ', r'\n  - ', text)
    
    return text.strip()

def extract_organ_and_procedure(report_text):
    """Extract organ and procedure from report text"""
    if not report_text or ';' not in report_text:
        return None, None
    
    procedure_part = report_text.split(';')[0].strip()
    if ',' not in procedure_part:
        return None, None
    
    organ = procedure_part.split(',')[0].strip()
    procedure = procedure_part.split(',', 1)[1].strip()
    return organ, procedure

def extract_diagnosis(report_text):
    """Extract diagnosis from text after newline"""
    if not report_text or '\n' not in report_text:
        return []
    
    # Get everything after the first newline
    diagnosis_text = report_text.split('\n', 1)[1].strip()
    if not diagnosis_text:
        return []
    
    # Handle both literal \n and actual newlines
    diagnosis_text = diagnosis_text.replace('\\n', '\n')
    
    # Split by double newlines to separate main diagnosis from notes
    major_parts = re.split(r'\n\s*\n', diagnosis_text)
    
    diagnoses = []
    for major_part in major_parts:
        major_part = major_part.strip()
        if not major_part:
            continue
        
        # Simple approach: split by digit + dot pattern anywhere
        if re.search(r'\d+\.', major_part):
            # Split using regex that finds digit.space pattern
            parts = re.split(r'\s+(?=\d+\.\s)', major_part)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # Remove number prefix
                clean_part = re.sub(r'^\d+\.\s*', '', part).strip()
                if clean_part:
                    diagnoses.append(clean_part)
        else:
            # Single diagnosis without numbering
            diagnoses.append(major_part)
    
    return diagnoses

def reconstruct_report(organ, procedure, diagnoses):
    """Reconstruct report from fixed components"""
    if not organ or not procedure:
        return ""
    
    # Build the procedure line
    report_lines = [f"{organ}, {procedure};"]
    
    # Add diagnoses
    if diagnoses:
        # Separate regular and Note) diagnoses
        regular_diagnoses = []
        note_diagnoses = []
        
        for diagnosis in diagnoses:
            if diagnosis.startswith("Note)"):
                note_diagnoses.append(diagnosis)
            else:
                regular_diagnoses.append(diagnosis)
        
        # Format regular diagnoses (numbered only if multiple)
        if len(regular_diagnoses) == 1:
            report_lines.append(f"  {regular_diagnoses[0]}")
        elif len(regular_diagnoses) > 1:
            for i, diagnosis in enumerate(regular_diagnoses, 1):
                report_lines.append(f"  {i}. {diagnosis}")
        
        # Add Note) diagnoses at the end (no numbering)
        if note_diagnoses:
            if regular_diagnoses:  # Add blank line if there were regular diagnoses
                report_lines.append("")
            for note_diagnosis in note_diagnoses:
                report_lines.append(f"  {note_diagnosis}")
    
    return '\n'.join(report_lines)

def main():
    # Load input data
    with open('../train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Remove specific ID entries
    data = [item for item in data if item.get('id') not in ['PIT_01_00213_01.tiff', 'PIT_01_04231_01.tiff']]
    
    results = []
    skipped_items = 0
    skipped_reasons = defaultdict(int)
    
    for item in data:
        original_report = item.get('report', '')
        
        if not original_report:
            skipped_items += 1
            skipped_reasons['no_report'] += 1
            continue
            
        # Extract components from original report
        organ, procedure = extract_organ_and_procedure(original_report)
        diagnoses = extract_diagnosis(original_report)
        
        # Fix typos
        if organ:
            organ = fix_organ_typos(organ)
        
        if procedure:
            procedure = fix_procedure_typos(procedure)
            
        # Skip items with invalid organ or procedure
        if not organ:
            skipped_items += 1
            skipped_reasons['invalid_organ'] += 1
            continue
            
        if not procedure:
            skipped_items += 1
            skipped_reasons['invalid_procedure'] += 1
            continue
            
        if diagnoses:
            # Handle special multi-diagnosis cases BEFORE applying fixes
            expanded_diagnoses = []
            for d in diagnoses:
                if "malignant lymphoma, extranodal marginal zone b cell lymphoma" in d.lower():
                    expanded_diagnoses.extend(["Malignant lymphoma", "Extranodal marginal zone B cell lymphoma of MALT type"])
                else:
                    expanded_diagnoses.append(d)
            
            # Apply typo fixes and normalization
            diagnoses = [normalize_spacing(fix_diagnosis_typos(d)) for d in expanded_diagnoses]
        else:
            skipped_items += 1
            skipped_reasons['no_diagnosis'] += 1
            continue
        
        # Create label field: organ + procedure + diagnosis words with underscores
        label_parts = [organ.lower()]

        # Add procedure as-is (with spaces, no punctuation)
        clean_procedure = re.sub(r'[^\w\s]', '', procedure)
        label_parts.append(clean_procedure.lower())

        # Process each diagnosis - connect words with underscores
        for diagnosis in diagnoses:
            clean_diagnosis = re.sub(r'[^\w\s]', '', diagnosis)
            diagnosis_words = clean_diagnosis.split()
            diagnosis_label = '_'.join([word.lower() for word in diagnosis_words])
            label_parts.append(diagnosis_label)

        label = '_'.join(label_parts)
        
        # Reconstruct the report from fixed components
        cleaned_report = reconstruct_report(organ, procedure, diagnoses)
        
        result_item = item.copy()
        result_item.update({
            'organ': organ,
            'procedure': procedure,
            'diagnosis': clean_diagnosis,
            'original': original_report,
            'report': cleaned_report,
            'label': label
        })
        results.append(result_item)
    
    # Save results
    with open('train.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    total = len(data)
    processed = len(results)
    
    print(f"Total items: {total}")
    print(f"Processed items: {processed}")
    print(f"Skipped items: {skipped_items}")
    print("\nSkip reasons:")
    for reason, count in skipped_reasons.items():
        print(f"  {reason}: {count}")

if __name__ == "__main__":
    main()