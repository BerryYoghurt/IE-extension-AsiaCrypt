chrome.contextMenus.create(
    {
        id: "decrypt",
        title: "Decrypt this image",
        contexts: ["image"]
    }
)


let decryptImage = async function(pwd, ctxt){
    console.log("Password inside background script: ", pwd);
    console.log("Inside decryptImage in background: ", ctxt);
    try {
        // const response = await fetch('http://localhost:8000/', {
        //     method: 'POST',
        //     headers: {
        //         'Content-Type': 'image/jpeg'
        //     },
        //     body: ctxt
        // });
        const formData = new FormData(); 
        formData.append('password', pwd);
        formData.append('image', ctxt);


        const response = await fetch('http://localhost:8000/', {
            method: 'POST',
            // headers: {
            //     'Content-Type': 'multipart/form-data'
            // },
            body: formData //ctxt
        });

        if (response.ok) {
            const responseBlob = await response.blob();
            // Process the response blob (e.g., display the image)
            console.log('Response received:', responseBlob);
            return responseBlob;
        } else {
            console.error('Failed to send POST request:', response.statusText);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

chrome.contextMenus.onClicked.addListener((item, tab)=>{
    if (item.menuItemId == "decrypt"){
        console.log("Inside listener")
        // First: get password with asynchronous call
        // let password = getPassword()
        // chrome.action.openPopup();

        // Second: get image
        // send message to content-script to send back its url
        chrome.tabs.sendMessage(tab.id, {request: "startDecrypt"})
                   .then((resp) => {
                        console.log("Response received in background: ", resp);
                        const password = resp.password
                        const imageBytes = resp.image
                        const bytes = Object.values(imageBytes);
                        // const bytes = Object.values(resp);
                        const blob = new Blob([new Uint8Array(bytes)], { type: 'image/jpeg' });
                        // return decryptImage(password, blob);
                        return decryptImage(password, blob);
                    })
                    .then((blob) => {
                        console.log("Response from server to arrayBuffer: ", blob);
                        return blob.arrayBuffer();
                    })
                    .then((arrayBuffer)=>{
                        chrome.tabs.sendMessage(tab.id, {request: "finishDecrypt", ptxt: new Uint8Array(arrayBuffer)});
                   });
        // download image
        // start decrypting
        // send message to content-script
    }
    console.log("Source of image in backgournd script: ", item.srcUrl);
    // console.log(tab);
    // console.log("Hello");
})