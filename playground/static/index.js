// function handleFileUpload(event) {
//     const file = event.target.files[0];
//     const reader = new FileReader();
//     reader.onload = function(e) {
//         const fileContent = e.target.result;
//         document.getElementById('display-content').style.display = 'block';
//         document.getElementById('fileContent').textContent = fileContent;
//     };
//     reader.readAsText(file);
// }

// var extractedText = document.getElementById("extracted-text").textContent.trim();
// if (extractedText !== "") {
//     document.getElementById("file-upload-section").style.display = "none";
//     document.getElementById("display-content").style.display = "block";
// }

// function dropHandler(ev) {
//     console.log("File(s) dropped");
  
//     // Prevent default behavior (Prevent file from being opened)
//     ev.preventDefault();
  
//     if (ev.dataTransfer.items) {
//       // Use DataTransferItemList interface to access the file(s)
//       [...ev.dataTransfer.items].forEach((item, i) => {
//         // If dropped items aren't files, reject them
//         if (item.kind === "file") {
//           const file = item.getAsFile();
//           console.log(`… file[${i}].name = ${file.name}`);
//         }
//       });
//     } else {
//       // Use DataTransfer interface to access the file(s)
//       [...ev.dataTransfer.files].forEach((file, i) => {
//         console.log(`… file[${i}].name = ${file.name}`);
//       });
//     }
// }
  
// function dragOverHandler(ev) {
//     console.log("File(s) in drop zone");

//     // Prevent default behavior (Prevent file from being opened)
//     ev.preventDefault();
// }
  
