To properly use this software, the proper steps must be taken otherwise it might not work properly.

The first step is to download the IDS 2018 Intrusion CSVs dataset https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv/data?select=02-20-2018.csv

Create a working directory (wherever) and create the following folders inside of it

Dataset
Checkpoints
DeployModel
ProcessedDataset
templates
static

styles.css should go in static, index.html should go in templates
The DLScript.py file should go in the working directory folder, and the etb_profiler.py should also go there.

Place all of the .csv files in the Dataset folder (if the download saves as a zip for example unzip it and place all the .csv files in the Dataset folder, do not put them in a folder inside, so for example it should be like this Dataset > 03-02-2018.csv for one of the files (same for rest, the IDS 2018 dataset should have 10 .csv files)

Now run the DLScript, and select option 1 or 2 (Process the normal or super simplified dataset), this will process the dataset and save everything to ProcessedDataset (this may take some time!).

Once it is finished now select 4, 5,6, or 7 (Train a new model)

Here you can select which model you’d like to train on the data, if you’d like to change how long it trains you can change the epoch variable in the script before running the script.

Once you’ve selected the model it should now start training, which may take a while (especially tab transformer can take a long time). After all the epoch are completed the finished model should be saved in the DeployModel folder.

Now you may exit the training script (option 20 or CONTROL C the terminal to terminate the script)

Now run the etb_profiler.py script

This script will create a server using flask and It should run it at http://127.0.0.1:5001.

Go to that IP on a web browser after the script is started, and here you can upload or copy paste Traffic flow data. 

If copying data directly from IDS 2018 dataset, make sure to add commas between the numbers 

Here are some examples from the dataset directly

Benign
53,17,2018-03-01 04:48:27,641,1,1,32,60,32,32,32,0,60,60,60,0,143525.741,3120.124805,641,0,641,641,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,8,1560.062402,1560.062402,32,60,41.33333333,16.16580754,261.3333333,0,0,0,0,0,0,0,0,1,62,32,60,0,0,0,0,0,0,1,32,1,60,-1,-1,0,8,0,0,0,0,0,0,0,0

DDOS attack LOIC UDP
80,17,2018-02-21 10:09:13,119851238,148637,0,4756384,0,32,32,32,0,0,0,0,0,39685.73107,1240.179096,806.340577,17284.93816,660561,0,120000000,806.340577,17284.93816,660561,0,0,0,0,0,0,0,0,0,0,1189096,0,1240.179096,0,32,32,32,0,0,0,0,0,0,0,0,0,0,0,32.00021529,32,0,0,0,0,0,0,0,148637,4756384,0,0,-1,-1,148636,8,0,0,0,0,0,0,0,0

DDOS ATTACK LOIC UDP

80,17,2018-02-21 10:10:01,119994216,116294,0,3721408,0,32,32,32,0,0,0,0,0,31013.22817,969.1633803,1031.826645,20038.6205,644494,0,120000000,1031.826645,20038.6205,644494,0,0,0,0,0,0,0,0,0,0,930352,0,969.1633803,0,32,32,32,0,0,0,0,0,0,0,0,0,0,0,32.00027516,32,0,0,0,0,0,0,0,116294,3721408,0,0,-1,-1,116293,8,0,0,0,0,0,0,0,0

DDOS ATTACK HOIC
80,6,2018-02-21 02:24:28,2820,2,0,0,0,0,0,0,0,0,0,0,0,0,709.2198582,2820,0,2820,2820,2820,2820,0,2820,2820,0,0,0,0,0,0,0,0,0,40,0,709.2198582,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,32738,-1,0,20,0,0,0,0,0,0,0,0

DDOS ATTACK HOIC
80,6,2018-02-21 02:24:28,15671,2,0,0,0,0,0,0,0,0,0,0,0,0,127.6242741,15671,0,15671,15671,15671,15671,0,15671,15671,0,0,0,0,0,0,0,0,0,40,0,127.6242741,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,32738,-1,0,20,0,0,0,0,0,0,0,0

BENIGN
51168,6,2018-02-21 02:24:28,4339,5,2,935,309,935,0,187,418.1447118,309,0,154.5,218.4959954,286702.0051,1613.274948,723.1666667,1479.066519,3729,7,4339,1084.75,2003.790138,4087,7,3729,3729,0,3729,3729,0,0,0,0,124,40,1152.339249,460.9356995,0,935,155.5,333.0087944,110894.8571,0,0,1,1,0,0,0,1,0,177.7142857,187,154.5,0,0,0,0,0,0,5,935,2,309,65535,32768,1,20,0,0,0,0,0,0,0,0


If you copy it directly from a CSV file it’ll likely not copy with commas

If you’d like to capture a Wireshark flow do the following

Start Wireshark and capture packets

Stop it once you’ve captured the packets you want to analyze, and select the packets you want to analyze (best to have enough for a full traffic flow)

Go to File > Export Specified Packets

Then at the bottom should be choices such as All Packets and Selected packets only, select selected packets only (unless of course you want all of them)

And then change Export as from pcapng to pcap

Now open up terminal and install the following program “cicflowmeter” 

Example command: pip3 install cicflowmeter 

After it installs, go to the directory with the pcap file and type this command cicflowmeter -f "TLSv Packet.pcap" -c test.csv

You can name the csv file whatever you want, and same with the pcap when you save it in Wireshark, just make sure to add “” if there is a space in the name.

Now go back to the Traffic Profiler website (or run it again if you closed it), select WireShark CICFlowMeter for Input Data Format and then either upload the CSV file (preferable) or paste the content.



