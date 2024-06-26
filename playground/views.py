from django.shortcuts import render, redirect
from .utils import *
from .nlp import summarize


def DocSum(request):
    summary_len = request.POST.get('summary_len')
    input_text = request.POST.get('input_text')
    summarized_text = ''
    uploaded_text = ''
    bart_summary= ''

    if 'uploaded_file' in request.FILES:
        uploaded_file = request.FILES['uploaded_file']
        if uploaded_file.name.endswith('.pdf'):
            uploaded_text = extract_text_from_pdf(uploaded_file)
            summarized_text, bart_summary = summarize(uploaded_text, int(summary_len))
        elif uploaded_file.name.endswith('.docx'):
            uploaded_text = extract_text_from_docx(uploaded_file)
            summarized_text, bart_summary = summarize(uploaded_text, int(summary_len))
        else:
            summarized_text, bart_summary = "Unsupported file format", "Unsupported file format"

    elif input_text is not None:
        uploaded_text = format_input_text(input_text)
        summarized_text, bart_summary = summarize(uploaded_text, int(summary_len))

    return render(
        request, 'index.html',
        {
            "uploaded_text": uploaded_text,
            "uploaded_text_len": len(uploaded_text),
            "summarized_text": summarized_text,
            "summarized_text_len": len(summarized_text),
            "bart_summary": bart_summary,
            "bart_summary_len": len(bart_summary),
        }
    )