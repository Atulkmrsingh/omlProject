import pdfplumber
from setcover import select_glossaries1
from probabilitysetcover import select_glossaries2
from facilityCover import select_glossaries3
from concaveOverModular import select_glossaries4

def isMachineReadable(pdf_file):
    try:
        pdf = pdfplumber.open(pdf_file)
    except:
        return

    for page_id in range(len(pdf.pages)):
        current_page = pdf.pages[page_id]
        words = current_page.extract_words()
        if(len(words)):
          break
    return len(words) > 0


# Returns preference order for english machine readable source, None otherwise
def get_preference_order(pdf_path, ocr_lang, src_lang, trans_lang, glossaries_path) :
    if src_lang != "en" or ocr_lang != "en" :
        return None

    if isMachineReadable(pdf_path) :
        #return select_glossaries1(pdf_path, src_lang, trans_lang, glossaries_path) #set cover
        return select_glossaries4(pdf_path, src_lang, trans_lang, glossaries_path)   #probability set cover
    return None
print(get_preference_order("BTP_report.pdf","en","en","hi","en-hi_acronym_dicts 2"))

    

