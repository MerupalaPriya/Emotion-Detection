import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import webbrowser
import os

# convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

# making folders
outer_names = ['test','train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data',outer_name), exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data',outer_name,inner_name), exist_ok=True)

# to keep count of each category
angry = 0
disgusted = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0

df = pd.read_csv('./fer2013.csv')
mat = np.zeros((48,48),dtype=np.uint8)
print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    # train
    if i < 28709:
        if df['emotion'][i] == 0:
            img.save('train/angry/im'+str(angry)+'.png')
            angry += 1
            webbrowser.open("https://www.youtube.com/watch?v=TdrL3QxjyVw&list=RDEMYGj5tu94_mNz6SrYkDD3_g&start_radio=1")
        elif df['emotion'][i] == 1:
            img.save('train/disgusted/im'+str(disgusted)+'.png')
            disgusted += 1
            webbrowser.open("https://www.youtube.com/watch?v=WKZO-CWeOVA&list=RDEMcce0hP5SVByOVCd8UWUHEA&start_radio=1")
        elif df['emotion'][i] == 2:
            img.save('train/fearful/im'+str(fearful)+'.png')
            fearful += 1
            webbrowser.open("https://www.youtube.com/watch?v=_WCD3Z9UmJ4&list=RDEMpkCBXSJsfVW_hwsQqdczqg&start_radio=1")
        elif df['emotion'][i] == 3:
            img.save('train/happy/im'+str(happy)+'.png')
            happy += 1
            webbrowser.open("https://www.youtube.com/watch?v=-WOonkg_ZCo&list=PLc6njbSPwsMzI9yibh9pkA1pnlPI12fEm&index=9")
        elif df['emotion'][i] == 4:
            img.save('train/sad/im'+str(sad)+'.png')
            sad += 1
            webbrowser.open("https://www.youtube.com/watch?v=-WOonkg_ZCo&list=PLc6njbSPwsMzI9yibh9pkA1pnlPI12fEm&index=4")
        elif df['emotion'][i] == 5:
            img.save('train/surprised/im'+str(surprised)+'.png')
            surprised += 1
            webbrowser.open("https://www.youtube.com/watch?v=ApXoWvfEYVU&list=RDApXoWvfEYVU&start_radio=1")
        elif df['emotion'][i] == 6:
            img.save('train/neutral/im'+str(neutral)+'.png')
            neutral += 1
            webbrowser.open("https://www.youtube.com/watch?v=CTFtOOh47oo&list=RDApXoWvfEYVU&index=2")

    # test
    else:
        if df['emotion'][i] == 0:
            img.save('test/angry/im'+str(angry_test)+'.png')
            angry_test += 1
            webbrowser.open("https://www.youtube.com/watch?v=TdrL3QxjyVw&list=RDEMYGj5tu94_mNz6SrYkDD3_g&start_radio=1")
        elif df['emotion'][i] == 1:
            img.save('test/disgusted/im'+str(disgusted_test)+'.png')
            disgusted_test += 1
            webbrowser.open("https://www.youtube.com/watch?v=WKZO-CWeOVA&list=RDEMcce0hP5SVByOVCd8UWUHEA&start_radio=1")
        elif df['emotion'][i] == 2:
            img.save('test/fearful/im'+str(fearful_test)+'.png')
            fearful_test += 1
            webbrowser.open("https://www.youtube.com/watch?v=_WCD3Z9UmJ4&list=RDEMpkCBXSJsfVW_hwsQqdczqg&start_radio=1")
        elif df['emotion'][i] == 3:
            img.save('test/happy/im'+str(happy_test)+'.png')
            happy_test += 1
            webbrowser.open("https://www.youtube.com/watch?v=-WOonkg_ZCo&list=PLc6njbSPwsMzI9yibh9pkA1pnlPI12fEm&index=9")
        elif df['emotion'][i] == 4:
            img.save('test/sad/im'+str(sad_test)+'.png')
            sad_test += 1
            webbrowser.open("https://www.youtube.com/watch?v=-WOonkg_ZCo&list=PLc6njbSPwsMzI9yibh9pkA1pnlPI12fEm&index=4")
        elif df['emotion'][i] == 5:
            img.save('test/surprised/im'+str(surprised_test)+'.png')
            surprised_test += 1
            webbrowser.open("https://www.youtube.com/watch?v=ApXoWvfEYVU&list=RDApXoWvfEYVU&start_radio=1")
        elif df['emotion'][i] == 6:
            img.save('test/neutral/im'+str(neutral_test)+'.png')
            neutral_test += 1
            webbrowser.open("https://www.youtube.com/watch?v=CTFtOOh47oo&list=RDApXoWvfEYVU&index=2")

print("Done!")