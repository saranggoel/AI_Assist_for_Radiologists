# AI Assist for Radiologists: A Smart, Automated Application Using Artificial Intelligence for COVID-19 Classification and Severity Prediction
## Pre-Requisites:
Note: Developed and tested on Windows 10 environment (preferred OS)
1. Python 3.7+ (https://www.python.org/downloads/release/python-370/)
2. Anaconda for Python (https://www.anaconda.com/products/individual)
3. PyCharm Community Edition with Anaconda Plugin (https://www.jetbrains.com/pycharm/download/other.html)

## Steps to Install and Run
1. Click the green pulldown "Code" button and select "Download ZIP".
2. Save the zipped file in a root directory.
3. Open the zipped file and extract it.
4. Within the project folder, download and save https://drive.google.com/file/d/1RpZx5bPvb3_-n2PB9gzZhPEgHOZ3mSoz/view?usp=sharing and https://drive.google.com/file/d/1Se_Mf9-0TqSzktKB7elU36uVq4Ut41EO/view?usp=sharing into the "models" folder.
5. Open PyCharm.
6. Click "File" and then "Open".
7. Select the project folder and click "OK".
8. Once the project has loaded on PyCharm, click "File" and the "Settings".
9. Click on the "Project: COVID-19_Classification_and_Severity_Prediction" tab and select "Project Interpreter".
10. On the top right, click the gear icon and select "Add...".
11. Select the "Conda Environment" tab and make sure "New environment" is selected. 
12. Leave the pre-assigned "Location" amd "Conda executable" values and make sure the "Python version" is 3.7.
13. Once the Conda environment is created, click "Apply" and then "OK".
14. Click "Add Configuration" (top-right of PyCharm window).
15. Click the "+" button and select Python.
16. Set the "Script path" as the project folder, and make sure the "Python interpreter" is set to the previously created environment.
17. Click "Apply" and then "OK".
18. Click "Terminal" (bottom-left of PyCharm window) and make sure that the environment is activated by checking that its name is displayed within parenthesis, at the front of the command line. If not, type "activate" and then the name of the previously created environment.
19. Make sure that you are within the project directory. If not, "cd" into it.
20. Run "pip install -r requirements.txt" to install the libraries used in the code. 
21. After the previous command has finished running, run "pip install git+https://github.com/JoHof/lungmask" to install the lungmask library.
22. After the previous command has finished running, go to https://pytorch.org/ and scroll down to "Install PyTorch".
23. Select "Stable" for the "PyTorch Build" and select your system's configurations. Run the command it provides in your PyCharm terminal.
24. On the "Project" panel, double click on "COVID-19 Classification and Severity Prediction Website.py" to open it up. Press "CTRL + R".
25. In the first blank, type "rootdir" and in the second blank, type in your project directory. Click the "Replace all" button.
26. Now that everything is ready, type "streamlit run "COVID-19 Classification and Severity Prediction Website.py"" in the PyCharm terminal. It should provide a link.
27. Click on the link and open it on your browser (has been tested on Google Chrome).
28. Once the website loads, click on the "Browse Files" tab. Go to the "Test Images" folder with your project directory. Go into any patient's folder. Select all the images and click "Open".
29. To view the uploaded CT scans, open the sidebar using the arrow on the top-left of the website. You can close it by clicking the "X" button as well. Additionally, enter in the patient's age, gender, and day's since suspected of COVID-19 according to the "Patient#Info.txt" file provided in the "Test Images" folder.
30. Once everything has been entered, click the "Run Classification and Severity Prediction Model on CT Volume" button. The website will run the classification and severity prediction models on the CT scans provided and output information pertaining to the patient's COVID-19 classification and severity. Note that this process may take a while, especially without a GPU. In real-world purposes, hospitals will be able to run the website in real-time, as they have access to much more powerful computers. Also, the website would be hosted on a public domain, allowing for easy access.
31. Once you are done running the patient's CT volume, you can reload the page and run the other patient's CT volumes. Otherwise, you can go back to PyCharm and press "CTRL + C" to stop the program.
