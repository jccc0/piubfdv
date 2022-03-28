import * as tf from "@tensorflow/tfjs";
import * as mobilenetModule from "@tensorflow-models/mobilenet";
import * as knnClassifier from "@tensorflow-models/knn-classifier";
import * as datajson from "./data.json";

console.log(datajson);


// Create the classifier.
let classifier = knnClassifier.create();
let dataset;

// Load mobilenet.
const mobilenet = await mobilenetModule.load();

// Add MobileNet activations to the model repeatedly for all classes.
// const img0 = tf.browser.fromPixels(document.getElementById('class0'));
// const logits0 = mobilenet.infer(img0, true);
// classifier.addExample(logits0, 0);

// const img1 = tf.browser.fromPixels(document.getElementById('class1'));
// const logits1 = mobilenet.infer(img1, true);
// classifier.addExample(logits1, 1);

// Make a prediction.
function makeprediction() {
    const x = tf.browser.fromPixels(document.getElementById('test'));
    const xlogits = mobilenet.infer(x, true);
    console.log('Predictions:');
    console.log(classifier.predictClass(xlogits));
}

function save(){
    // Stringify the dataset
    dataset = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data])=>[label, Array.from(data.dataSync()), data.shape]) );
    // Save the dataset
    download(dataset, "data.json", "text/plain");
    
}

async function load(){
     // Load the dataset 
     // Add to a new classifier
     classifier = knnClassifier.create();
     classifier.setClassifierDataset( Object.fromEntries( datajson.map(([label, data, shape])=>[label, tf.tensor(data, shape)]) ) );
}

function download(content, fileName, contentType) {
    var a = document.createElement("a");
    var file = new Blob([content], { type: contentType });
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
  }


let ss = document.getElementById("save");
let ll = document.getElementById("load");
let mm = document.getElementById("predict");

ss.onclick = function(){save()};
ll.onclick = function(){load()};
mm.onclick = function(){makeprediction()};


