const express = require('express');
const {spawn} = require('child_process');
const app = express();
const fs = require('fs');
const port = 3000
var proceedingVisitor = false;
app.use(express.json({inflate: true, strict: false, type: () => { return true; } }));
app.post('/rec', async (req, res) => {
	var dataToSend;
	// spawn new child process to call the python script
	const python = spawn('python3', ['userTrackFetch.py', req.body.token], { timeout: 1000 * 30 });
	// collect data from script
	python.stdout.on('data', function (data) {
		console.log('Pipe data from python script ...');
		dataToSend = data.toString();
	});
	// in close event we are sure that stream from child process is closed
	python.on('close', (code) => {
		console.log(`child process close all stdio with code ${code}`);
		// send data to browser
		res.setHeader('Access-Control-Allow-Origin', '*');
		res.setHeader('Content-Type', 'application/json');
		res.send(JSON.stringify(dataToSend.replace(/(\r\n|\n|\r)/gm, "").replace(/'/g, "\"")));
	});

	while(proceedingVisitor){
		await new Promise(resolve=>{
		setTimeout(()=>{
			resolve();
		},5)
		})
	}
	proceedingVisitor = true;
	let result = fs.readFileSync('numvisitors.txt');  
	result++;
	fs.writeFileSync('numvisitors.txt', String(result));
	proceedingVisitor = false;
	while(proceedingPage){
		await new Promise(resolve=>{
		setTimeout(()=>{
			resolve();
		},5)
		})
	}
	proceedingPage = true;
	result = fs.readFileSync('numpages.txt');  
	result++;
	fs.writeFileSync('numpages.txt', String(result));
	proceedingPage = false;
})
var proceedingSave = false;
app.get('/save', async (req, res) => {
	while(proceedingSave){
		await new Promise(resolve=>{
		setTimeout(()=>{
			resolve();
		},5)
		})
	}
	proceedingSave = true;
	let result = fs.readFileSync('numsaves.txt');  
	result++;
	fs.writeFileSync('numsaves.txt', String(result));
	proceedingSave = false;
	res.setHeader('Access-Control-Allow-Origin', '*');
	res.send("");
})
var proceedingPage = false;
app.get('/page', async (req, res) => {
	while(proceedingPage){
		await new Promise(resolve=>{
		setTimeout(()=>{
			resolve();
		},5)
		})
	}
	proceedingPage = true;
	let result = fs.readFileSync('numpages.txt');  
	result++;
	fs.writeFileSync('numpages.txt', String(result));
	proceedingPage = false;
	res.setHeader('Access-Control-Allow-Origin', '*');
	res.send("");
})
app.listen(port, () => console.log(`Example app listening on port ${port}!`))