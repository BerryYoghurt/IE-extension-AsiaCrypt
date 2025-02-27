var targetImg = null;

var overlayDecryptedImage = (img) => {
    // listen on message
    var decryptedImageSource = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Almonds_-_in_shell%2C_shell_cracked_open%2C_shelled%2C_blanched.jpg/435px-Almonds_-_in_shell%2C_shell_cracked_open%2C_shelled%2C_blanched.jpg";
    // inject image
    img.src = decryptedImageSource;
    img.srcset = decryptedImageSource;
}


document.addEventListener("contextmenu", (event) => {
    console.log("Content script got an event");
    if (event.target instanceof HTMLImageElement){
        targetImg = event.target;
        console.log("The target is an image")
        console.log(event.target);
    }
}, true);


// downloads the image from the SNS and sends it to the background script
let downloadImage = function(url, sendResponse){
    const password = prompt("Enter password for this image:")
    
    return fetch(url,{
        credentials: 'include'
    })
    .then((response) => response.blob())
    .then((blob) => blob.arrayBuffer())
    .then((arrayBuffer) => {
        console.log("Downloading image in content-script: ", arrayBuffer);
        // console.log("Inside downloadImage in content-script: ", myBlob.bytes);
        sendResponse({"password": password, "image": new Uint8Array(arrayBuffer)});
        // sendResponse(new Uint8Array(arrayBuffer));
    });
}

chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
    if(message.request == "startDecrypt") {
        // console.log("Target image url: ", targetImg.url);
        // console.log("Target image src: ", targetImg.src);
        console.log("Content script received decrypt message from backgournd script")
        downloadImage(targetImg.src, sendResponse);
    }
    else if (message.request == "finishDecrypt"){
        // TODO make the extension show a popup telling the user that it's loading.
        console.log("Content script received decrypted image from backgournd script")
        const bytes = Object.values(message.ptxt);
        console.log("Bytes received in content-script: ", bytes);
        const blob = new Blob([new Uint8Array(bytes)], { type: 'image/jpeg' });
        console.log("Blob received in content-script: ", blob);
        const objectURL = URL.createObjectURL(blob);
        targetImg.src = objectURL;
        targetImg.srcset = objectURL;
    }
    // makes sendResponse asynchronous
    return true;
});
