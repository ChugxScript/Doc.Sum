
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = (textarea.scrollHeight) + 'px';
}

function fileChangeHandler(event) {
    const fileInput = event.target;
    const fileList = fileInput.files;

    if (fileList.length > 1) {
        alert("Please select only one file.");
        fileInput.value = '';
        return;
    }

    if (fileList.length === 1) {
        const fileName = fileList[0].name;
        const uploadFileDiv = fileInput.closest('.upload-file');
        const dropText = uploadFileDiv.querySelector('p:last-of-type');
        dropText.textContent = fileName;

        // Clear textarea
        const textarea = uploadFileDiv.closest('.input-container').querySelector('textarea');
        textarea.value = '';
    }
}

function dropHandler(event) {
    event.preventDefault();

    const fileInput = event.currentTarget.querySelector('input[type="file"]');
    const fileList = event.dataTransfer.files;

    if (fileList.length > 1) {
        alert("Please drop only one file.");
        return;
    }

    if (fileList.length === 1) {
        fileInput.files = fileList;
        fileChangeHandler({ target: fileInput });
    }
}

function dragOverHandler(event) {
    event.preventDefault();
}

document.addEventListener('DOMContentLoaded', function() {
    const removeBtns = document.querySelectorAll('.remove_file_btn');
    
    removeBtns.forEach(function(removeBtn) {
        removeBtn.addEventListener('click', function() {
            const fileInput = this.parentElement.querySelector('input[type="file"]');
            fileInput.value = ''; // Clear the file input
            const uploadFileDiv = fileInput.closest('.upload-file');
            const dropText = uploadFileDiv.querySelector('p:last-of-type');
            dropText.textContent = "NO FILE CHOSEN";
        });
    });

    const docsum_logo = document.getElementById('docsum_logo');
    docsum_logo.setAttribute('src', 'https://media1.tenor.com/m/Z_KEDm9F_hQAAAAC/document-office.gif');
});

