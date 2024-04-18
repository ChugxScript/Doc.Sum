from django.shortcuts import render, redirect
from .forms import FileUploadForm
from .utils import extract_text_from_pdf, extract_text_from_docx


def DocSum(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            # Read the uploaded file and extract text based on its type
            if uploaded_file.name.endswith('.pdf'):
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith('.docx'):
                extracted_text = extract_text_from_docx(uploaded_file)
            else:
                extracted_text = "Unsupported file format"
            return render(request, 'index.html', {'form': form, 'extracted_text': extracted_text})
    else:
        form = FileUploadForm()
    return render(request, 'index.html', {'form': form})







# from django.http import JsonResponse
# from rest_framework.views import APIView

# class ReactView(APIView): 

#     def get(self, request):
#         # Handle GET request logic here
#         return render(request, 'index.html')

#     def post(self, request): 
#         if 'file' in request.FILES:
#             uploaded_file = request.FILES['file']
#             file_content = uploaded_file.read().decode('utf-8')
#             return JsonResponse({"file_content": file_content})
#         else:
#             return JsonResponse({"error": "No file uploaded."}, status=400)