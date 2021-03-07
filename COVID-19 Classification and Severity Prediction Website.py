# This file creates the website and implements all models to create final product. Follow all steps in the ReadMe.md file
# to execute the code.

from __future__ import print_function
import keras
import streamlit as st
import cv2
from lungmask import mask
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import numpy as np
import os
import imageio
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import test_dataset
import io
from PIL import Image as pil_image
from keras_retinanet import layers
import keras.backend as k
from PIL import Image
import keras_retinanet
import torch

torch.cuda.empty_cache()

k.clear_session()

st.set_page_config(page_title="COVID-19 Web App", initial_sidebar_state="collapsed", layout="wide") #sets website configuration


#first function to run
def main():
    st.title("A Machine Learning Model for COVID-19 Prediction and Diagnosis Using Computed Tomography of Chest") #title of website
    inputs = user_input_features()
    df = inputs[0]
    age = inputs[1]
    gender = inputs[2]
    days = inputs[3]
    length = len(df)
    if length > 0:
        # following code displays sidebar of original uploaded images
        st.sidebar.subheader("List of Patient's CT Scans:")
        directory = 'rootdir/Images/'
        savedirectory = 'rootdir/Images - Refined/'
        x = 0
        for files in os.listdir(directory):
            path = directory + files
            save_path = savedirectory + files
            with open(path, 'rb') as f:
                tif = pil_image.open(io.BytesIO(f.read()))
            array = np.array(tif)
            max_val = np.amax(array)
            normalized = (array / max_val)
            im = pil_image.fromarray(normalized)
            im.save(save_path)
            st.sidebar.write("Filename: ", df[x][1])
            st.sidebar.image(normalized, width=270)
            st.sidebar.write(" ")
            x += 1
        st.write(" ")
        st.write(" ")
    else:
        st.sidebar.subheader("The patient's list of CT scans will be listed here once they are uploaded.")

    if st.button('Run Classification and Severity Prediction Models On CT Volume'):
        with st.spinner("Process ongoing"):
            prob = run(df)
            rg_function(prob[0], prob[1], prob[2], prob[3])
            segment(df)
            regions()
            area(df)


def delete_files(folder):
    # deletes files in current folders
    import os, shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try :
            if os.path.isfile(file_path) or os.path.islink(file_path) :
                os.unlink(file_path)
            elif os.path.isdir(file_path) :
                shutil.rmtree(file_path)
        except Exception as e :
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def user_input_features():
    # provides widgets for user to upload files and enter patient data
    uploaded_files = st.file_uploader("Step 1: Choose Patient's CT Scan File(s)", accept_multiple_files=True, type=["tif"])
    names = []
    for i in uploaded_files:
        names.append(i.name)
    resultImages = list(zip(uploaded_files, names))
    resultImages.sort(key = lambda x:x[1])
    for i in resultImages:
        im = Image.open(i[0])
        arr = np.array(im)
        imageio.imsave('rootdir/Images/' + i[1], arr)
    st.write("To view uploaded images, open the sidebar (located toward the left of the screen).")
    col1, col2, col3 = st.beta_columns(3)
    st.write(" ")
    user_input = col1.number_input("Step 2: Enter Patient's Age", max_value=120, min_value=0, value=65)
    gender = col2.selectbox("Step 3: Choose Patient's Gender", ["Male", "Female", "Other"])
    days = col3.number_input("Step 4: Enter Day's Since Suspected of COVID-19", max_value=20, min_value=0, value=3)
    return resultImages, user_input, gender, days


def run(df):
    # runs the COVID-19 classification model on patient's CT scans
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #set to "-1" - this works to run on cpu only; set to "0" to run on cuda
    trained_models = ["rootdir/models/ResNet50V2-FPN-fold1-03-0.9485.hdf5"]

    for trn_model in trained_models:
      k.clear_session()
      custom_object={'UpsampleLike': keras_retinanet.layers._misc.UpsampleLike}
      netpath=trn_model
      net=keras.models.load_model(netpath, custom_objects=custom_object) #load model

      img_count = 0
      patientnum_last = 0
      pred_ind_sum=0
      pred_ind_prob_sum=0
      pred_avg=0
      prob_avg=0
      pred_ind_patient = 0

      length = len(df)
      image_threshold = 0.2
      patient_threshold = 0.3

      prob_list = []

      middlehalf = (len(df)) / 3
      for i in range(int(middlehalf), int(len(df) - middlehalf)):
          im = Image.open(df[i][0])
          img = np.array(im)
          pred_ind_prob = net.predict(np.expand_dims(np.expand_dims(img, axis=0), axis=3))[0]
          img_count += 1
          if (pred_ind_prob[0] >= image_threshold) :
              pred_ind = 0  # covid
          else :
              pred_ind = 1  # normal

          pred_ind_sum+=pred_ind
          pred_ind_prob_sum+=pred_ind_prob[0]
          pred_avg = round(pred_ind_sum / img_count, 3)
          prob_avg = round(pred_ind_prob_sum / img_count, 3)
          frac_img_covid = 1 - pred_avg
          prob_list.append(pred_ind_prob[0])

      if (frac_img_covid >= patient_threshold) :
          pred_ind_patient = 0
          pred_ind_patient_label = "COVID-19"
      else :
          pred_ind_patient = 1
          pred_ind_patient_label = "NORMAL"
      st.write("Number of CT Scans: ", length)
      st.write("Number of CT Scans Used for Classification: ", img_count)
      return length, prob_list, frac_img_covid, pred_ind_patient_label


def rg_function(length, newprob, fraction, prediction):
    # creates statistics for the user to see, and generates plots/graphs as well based on results
    middlehalf = (length)/3
    m = round(statistics.mean(newprob),4)
    s= round(np.std(newprob), 4)
    v= round(np.var(newprob, ddof=1), 4)
    q = [np.quantile(newprob, 0.25), np.quantile(newprob, 0.5), np.quantile(newprob, 0.75)]
    q_lower =round(q[0],4)
    q_median =round(q[1],4)
    q_upper=round(q[2],4)
    newprob_min=round(min(newprob),4)
    newprob_max = round(max(newprob), 4)

    x1 = []

    for i in range(len(newprob)):
        x1.append(i + middlehalf)
    y1 = newprob

    # scatter plot
    fig1 = plt.figure(figsize=(6, 5))
    ax1 = plt.axes()
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.scatter(x1, y1)
    ax1.set_title('Probability of COVID-19 vs. CT Scan Number Scatter Plot')
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True)
    ax1.set_xlabel('CT Scan Number')
    ax1.set_ylabel('Probability of COVID-19')

    image_threshold = 0.2
    patient_threshold = 0.3
    st.write("Image Threshold Value: ", image_threshold, " Patient Threshold Value: ", patient_threshold)
    st.write("COVID-19 Probability Statistics -- ", " Mean: ",m," Median: ", q_median," Standard Deviation: ", s, " Variance: ",v)
    st.write("Percentage of CT Scans Predicted with COVID-19: ", fraction*100)

    st.write("Box Plot Analysis -- ", " Minimum: ", newprob_min, " Lower Quartile Range: ", newprob_min, "-", q_lower, " Interquartile Range: ", q_lower, "-", q_upper," Upper Quarter Range: ", q_upper, "-", newprob_max, " Maximum: ", newprob_max)

    # box plot
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    axs.ticklabel_format(useOffset=False, style='plain')
    all_data = newprob
    axs.boxplot(all_data, vert=False, showfliers=False, whis='range')
    axs.set_title('CT Scan Image Box Plot Analysis')

    axs.set_yticks([])
    axs.xaxis.grid(True)
    axs.set_xlabel('Probability of COVID-19')
    axs.set_ylabel('')

    def get_img_from_fig(fig, dpi=180) :
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    col1, col2 = st.beta_columns(2)

    plot_img_np = get_img_from_fig(fig1)
    col1.image(plot_img_np, use_column_width=True)

    plot_img_np = get_img_from_fig(fig)
    col2.image(plot_img_np, use_column_width=True)
    st.markdown(
        f"<h6 style='text-align: center; color: black;'>Note: The middle-third of patient volume was used for classification purposes, as these images provided the best view of the lungs.</h6>",
        unsafe_allow_html=True)
    st.markdown(
        f"<h1 style='text-align: center; color: black;'><b>PATIENT CLASSIFICATION: {prediction}</b></h1>",
        unsafe_allow_html=True)


def segment(files):
    # creates abnormality masks
    dirr = 'rootdir/Images - Refined/'
    saveddir = 'rootdir/Images - JPG/'

    for file in os.listdir(dirr):
        im = Image.open(dirr + file)
        img_arr = np.array(im)
        for index, value in enumerate(img_arr):
            for ind, val in enumerate(value):
                img_arr[index][ind] *= 255
        new_im = Image.fromarray(img_arr)
        new_im = new_im.convert("L")
        file = str(file).rstrip(".tif")
        new_im.save(saveddir + file + '.jpg', 'JPEG')

    model = Network()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]) # uncomment it if you have multiply GPUs.
    model.load_state_dict(torch.load('rootdir/Snapshots/save_weights/Inf-Net/Inf-Net-100.pth', map_location={'cuda:1': 'cuda:0'}))
    model.cuda()
    model.eval()

    image_root = 'rootdir/Images - JPG/'
    test_loader = test_dataset(image_root, 512)

    y = 0
    for i in range(test_loader.size):
        image, name = test_loader.load_data()

        image = image.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)
        #
        res = lateral_map_2
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave('rootdir/Images - Masks/' + files[y][1].rstrip(".tif") + '.jpg', res)
        mask = res >= 0.05
        res[mask] = 1
        rmask = res < 0.05
        res[rmask] = 0
        cv2.imwrite('rootdir/Images - Whitemasks/' + files[y][1].rstrip(".tif") + '.tif', res)
        y += 1


def regions():
    # creates lung lobe segmentation masks
    for files in os.listdir('rootdir/Images/'):
        i = sitk.ReadImage('rootdir/Images/' + files)
        i_np = sitk.GetArrayFromImage(i)
        i_np = i_np[None, ::]
        maxnum = i_np.max()
        i_np = ((i_np/maxnum) * 2000) - 1000
        i_hu = sitk.GetImageFromArray(i_np)
        segmentation = mask.apply_fused(i_hu)
        result_out = sitk.GetImageFromArray(segmentation)
        sitk.WriteImage(result_out, 'rootdir/Images - RegionTif/' + files)

    for imgs in os.listdir('rootdir/Images - RegionTif/'):
        before = Image.open('rootdir/Images - RegionTif/' + imgs)
        gray = np.array(before)
        backtorgb = np.stack((gray,) * 3, axis=-1)
        color = np.array(backtorgb)
        for index, value in enumerate(gray):
            for ind, val in enumerate(value):
                if val == 1:
                    color[index][ind] = (0, 0, 255)  # red
                if val == 2:
                    color[index][ind] = (0, 255, 0)  # lime
                if val == 3:
                    color[index][ind] = (255, 0, 255)  # magenta
                if val == 4:
                    color[index][ind] = (0, 255, 255)  # cyan
                if val == 5:
                    color[index][ind] = (255, 165, 0)  # orange
        cv2.imwrite('rootdir/Images - Regionscolored/' + imgs, color)


def area(names):
    # brings everything together to calculate total lung infection and final severity score
    totalab = 0
    finlobe1 = 0
    finlobe2 = 0
    finlobe3 = 0
    finlobe4 = 0
    finlobe5 = 0
    fincross1 = 0
    fincross2 = 0
    fincross3 = 0
    fincross4 = 0
    fincross5 = 0
    print('\n\n\nAbnormality Area per Slice: ')
    for imgfile in os.listdir('rootdir/Images - Whitemasks/'):
        file = Image.open('rootdir/Images - Whitemasks/' + imgfile)
        img = np.array(file)
        counter = 0
        for index, value in enumerate(img):
            for ind, val in enumerate(value):
                if val == 1:
                    counter += 1
        print(counter)
        totalab += counter
    print('\n\n\nLobe Area per Slice: ')
    for file in os.listdir('rootdir/Images - RegionTif/'):
        im = Image.open('rootdir/Images - RegionTif/' + file)
        img_arr = np.array(im)
        lobe1 = 0
        lobe2 = 0
        lobe3 = 0
        lobe4 = 0
        lobe5 = 0
        for index, value in enumerate(img_arr):
            for ind, val in enumerate(value):
                if val == 1:
                    lobe1 += 1
                if val == 2:
                    lobe2 += 1
                if val == 3:
                    lobe3 += 1
                if val == 4:
                    lobe4 += 1
                if val == 5:
                    lobe5 += 1
        print(lobe1, lobe2, lobe3, lobe4, lobe5)
        finlobe1 += lobe1
        finlobe2 += lobe2
        finlobe3 += lobe3
        finlobe4 += lobe4
        finlobe5 += lobe5
    print('\n\n\nCross (Abnormality in Lobe) Area per Slice: ')
    z = 0
    for imgs in os.listdir('rootdir/Images - Whitemasks/'):
        cross1 = 0
        cross2 = 0
        cross3 = 0
        cross4 = 0
        cross5 = 0
        masks = Image.open('rootdir/Images - Whitemasks/' + imgs)
        region = Image.open('rootdir/Images - RegionTif/' + imgs)
        finmasks = np.array(masks)
        finregion = np.array(region)
        finarray = finregion
        for index, value in enumerate(finmasks):
            for ind, val in enumerate(value):
                finarray[index][ind] = finmasks[index][ind] * finregion[index][ind]
        for index, value in enumerate(finarray):
            for ind, val in enumerate(value):
                if val == 1:
                    cross1 += 1
                if val == 2:
                    cross2 += 1
                if val == 3:
                    cross3 += 1
                if val == 4:
                    cross4 += 1
                if val == 5:
                    cross5 += 1
        cv2.imwrite('rootdir/Images - Cross Area/' + names[z][1], finarray)
        print(cross1, cross2, cross3, cross4, cross5)
        fincross1 += cross1
        fincross2 += cross2
        fincross3 += cross3
        fincross4 += cross4
        fincross5 += cross5
        z += 1
    for imgs in os.listdir('rootdir/Images - Cross Area/'):
        before = Image.open('rootdir/Images - Cross Area/' + imgs)
        gray = np.array(before)
        backtorgb = np.stack((gray,) * 3, axis=-1)
        color = np.array(backtorgb)
        for index, value in enumerate(gray):
            for ind, val in enumerate(value):
                if val == 1:
                    color[index][ind] = (0, 0, 255)  # red
                if val == 2:
                    color[index][ind] = (0, 255, 0)  # lime
                if val == 3:
                    color[index][ind] = (255, 0, 255)  # magenta
                if val == 4:
                    color[index][ind] = (0, 255, 255)  # cyan
                if val == 5:
                    color[index][ind] = (255, 165, 0)  # orange
        cv2.imwrite('rootdir/Images - CrossAreaColored/' + imgs, color)
    sslobe1 = fincross1 / finlobe1
    sslobe2 = fincross2 / finlobe2
    sslobe3 = fincross3 / finlobe3
    sslobe4 = fincross4 / finlobe4
    sslobe5 = fincross5 / finlobe5
    print('\n\n\nFinal Abnormality Area:   ', totalab)
    print('Final Lobe Area:   ', finlobe1, finlobe2, finlobe3, finlobe4, finlobe5)
    print('Final Cross (Abnormality in Lobe) Area:   ', fincross1, fincross2, fincross3, fincross4, fincross5)
    print('Severity score:   ', sslobe1, sslobe2, sslobe3, sslobe4, sslobe5)
    print('Patient Diagnosis Complete!')

    regcolored = []
    crossareacolored = []

    for files in os.listdir('rootdir/Images - JPG/'):
        file = Image.open('rootdir/Images - JPG/' + files)
        img = np.array(file)
        regcolored.append(img)
    for files in os.listdir('rootdir/Images - CrossAreaColored/'):
        file = Image.open('rootdir/Images - CrossAreaColored/' + files)
        img = np.array(file)
        crossareacolored.append(img)

    col1, col2 = st.beta_columns(2)
    image_iterator = paginator(regcolored)
    indices_on_page, images_on_page = map(list, zip(*image_iterator))
    col1.markdown("<h3 style='text-align: left; color: black;'><b>Original CT Slices:</b></h3>", unsafe_allow_html=True)
    col1.image(images_on_page, width=60, caption=indices_on_page)

    image_iterator = paginator(crossareacolored)
    indices_on_page, images_on_page = map(list, zip(*image_iterator))
    col2.markdown("<h3 style='text-align: left; color: black;'><b>Abnormality Area in Lobes of CT Slices:</b></h3>",
                  unsafe_allow_html=True)
    col2.image(images_on_page, width=60, caption=indices_on_page)

    lobeper1 = np.round(((fincross1 / finlobe1) * 100), 3)
    lobeper2 = np.round(((fincross2 / finlobe2) * 100), 3)
    lobeper3 = np.round(((fincross3 / finlobe3) * 100), 3)
    lobeper4 = np.round(((fincross4 / finlobe4) * 100), 3)
    lobeper5 = np.round(((fincross5 / finlobe5) * 100), 3)

    def determineserscore(x):
        if x == 0:
            return (0, 'NORMAL')
        elif 0 < x <= 5:
            return (1, 'MILD')
        elif 5 < x <= 25:
            return (2, 'MILD-MODERATE')
        elif 25 < x <= 50:
            return (3, 'MODERATE')
        elif 50 < x <= 75:
            return (4, 'SEVERE')
        else:
            return (5, 'CRITICAL')

    pd.options.display.max_rows = 40

    coldict = {'Lobe 1 Area (LU)': 'red', 'Lobe 2 Area (LL)': 'springgreen', 'Lobe 3 Area (RU)': 'magenta',
               'Lobe 4 Area (RM)': 'yellow',
               'Lobe 5 Area (RL)': 'deepskyblue',
               'Abnormality Area in Lobe 1 (LU)': 'red', 'Abnormality Area in Lobe 2 (LL)': 'springgreen',
               'Abnormality Area in Lobe 3 (RU)': 'magenta'
        , 'Abnormality Area in Lobe 4 (RM)': 'yellow', 'Abnormality Area in Lobe 5 (RL)': 'deepskyblue'}

    rowdict = {'LEFT UPPER': 'red', 'LEFT LOWER': 'springgreen', 'RIGHT UPPER': 'magenta',
               'RIGHT MIDDLE': 'yellow', 'RIGHT LOWER': 'deepskyblue'}

    def highlight_cols(s, coldict):
        if s.name in coldict.keys():
            return ['background-color: {}'.format(coldict[s.name])] * len(s)
        return [''] * len(s)

    def highlight_rows(row, rowdict):
        if row.name in rowdict.keys():
            return ['background-color: {}'.format(rowdict[row.name])] * len(row)
        return [''] * len(row)

    dataframe = pd.DataFrame(
        data=([finlobe1, finlobe2, finlobe3, finlobe4, finlobe5, fincross1, fincross2, fincross3
                  , fincross4, fincross5],
              [' ', ' ', ' ', ' ', ' ', str(lobeper1) + '%', str(lobeper2) + '%',
               str(lobeper3) + '%', str(lobeper4) + '%', str(lobeper5) + '%']),
        columns=['Lobe 1 Area (LU)', 'Lobe 2 Area (LL)', 'Lobe 3 Area (RU)', 'Lobe 4 Area (RM)',
                 'Lobe 5 Area (RL)',
                 'Abnormality Area in Lobe 1 (LU)', 'Abnormality Area in Lobe 2 (LL)',
                 'Abnormality Area in Lobe 3 (RU)'
            , 'Abnormality Area in Lobe 4 (RM)', 'Abnormality Area in Lobe 5 (RL)'],
        index=['Total Area', 'Lobe % Infection'])
    st.table(dataframe.style.apply(highlight_cols, coldict=coldict))

    st.markdown("<h3 style='text-align: center; color: black;'><b>Lobe and Cumulative Severity Score Summary:</b></h3>",
                unsafe_allow_html=True)
    col3, col4 = st.beta_columns(2)
    dataframe1 = pd.DataFrame(
        data=([str(lobeper1) + '%', determineserscore(lobeper1)[0], determineserscore(lobeper1)[1]],
              [str(lobeper2) + '%', determineserscore(lobeper2)[0], determineserscore(lobeper2)[1]],
              [str(lobeper3) + '%', determineserscore(lobeper3)[0], determineserscore(lobeper3)[1]],
              [str(lobeper4) + '%', determineserscore(lobeper4)[0], determineserscore(lobeper4)[1]],
              [str(lobeper5) + '%', determineserscore(lobeper5)[0], determineserscore(lobeper5)[1]]),
        columns=['LOBE % INFECTION', 'LOBE SEVERITY SCORE', 'LOBE SEVERITY DESCRIPTION'],
        index=['LEFT UPPER', 'LEFT LOWER', 'RIGHT UPPER', 'RIGHT MIDDLE', 'RIGHT LOWER'])
    col3.table(dataframe1.style.apply(highlight_rows, rowdict=rowdict, axis=1))

    lobeseverityscore = determineserscore(lobeper1)[0] + determineserscore(lobeper2)[0] + determineserscore(lobeper3)[
        0] + determineserscore(lobeper4)[0] + determineserscore(lobeper5)[0]

    def desc(lss):
        if lss == 0:
            return 'NORMAL'
        elif 1 <= lss <= 5:
            return 'MILD'
        elif 6 <= lss <= 10:
            return 'MILD-MODERATE'
        elif 11 <= lss <= 15:
            return 'MODERATE'
        elif 16 <= lss <= 20:
            return 'SEVERE'
        else:
            return 'CRITICAL'

    col4.markdown(
        f"<h1 style='text-align: center; color: black;'><b>CUMULATIVE SEVERITY SCORE:<br>{lobeseverityscore} [{desc(lobeseverityscore)}]</b></h1>",
        unsafe_allow_html=True)


def paginator(items, items_per_page=53):
    items = list(items)
    page_number = 0
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), min_index, max_index)


if __name__ == "__main__":
    delete_files('rootdir/Images/')
    delete_files('rootdir/Images - Refined/')
    delete_files('rootdir/Images - Cross Area/')
    delete_files('rootdir/Images - CrossAreaColored/')
    delete_files('rootdir/Images - JPG/')
    delete_files('rootdir/Images - Masks/')
    delete_files('rootdir/Images - Regionscolored/')
    delete_files('rootdir/Images - RegionTif/')
    delete_files('rootdir/Images - Whitemasks/')
    main()
