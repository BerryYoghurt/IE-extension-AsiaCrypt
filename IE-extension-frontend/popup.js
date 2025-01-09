// function popup() {
//     const userInput = document.getElementById('userInput').value;
//     chrome.tabs.query({currentWindow: true, active: true}, function (tabs){
//         const activeTab = tabs[0];
//         chrome.tabs.sendMessage(activeTab.id, {"message": "sendPassword", "password": userInput});
//     });
// }

// document.getElementById('submit').addEventListener('click', popup);