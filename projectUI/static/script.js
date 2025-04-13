document.getElementById("fileInput").addEventListener("change", function(event) {
            let reader = new FileReader();
            reader.onload = function() {
                let output = document.getElementById("imagePreview");
                output.src = reader.result;
                output.style.display = "block";
            };
            reader.readAsDataURL(event.target.files[0]);
        });