# EASYOCR Trainer

EasyOCR training engine cloned from: git clone https://github.com/JaidedAI/EasyOCR.git

[Trainer File](trainer/train.py)

Dataset:

[Training Data](EasyOCR/trainer/all_data/train_labels.txt)

[Test Data](EasyOCR/trainer/all_data/test_labels.txt)

COMMAND: python train.py --train_data all_data/en --valid_data all_data/en --character "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -" --batch_size 2 --num_iter 300 --valInterval 100 --workers 0 --sensitive --lr 0.0001 --optim adam --decode greedy

### TESTING RESULTS:

super().**init**(

iter 0 | loss 45.1389

iter 10 | loss 4.4470

iter 20 | loss 4.3817

iter 30 | loss 3.6736

iter 40 | loss 3.8403

iter 50 | loss 3.7888

iter 60 | loss 4.8160

iter 70 | loss 3.3449

iter 80 | loss 3.5571

iter 90 | loss 3.6867

iter 100 | loss 2.9576

training time: 297.15257382392883

[100/300] Train loss: 5.25205, Valid loss: 3.71828, Elapsed_time: 297.15462

---

## Ground Truth | Prediction | Confidence Score & T/F

Soriano | | 0.0065 False

VALENTINE | | 0.0117 False

---

validation time: 56.13448619842529

iter 110 | loss 3.5413

iter 120 | loss 3.1034

iter 130 | loss 5.2702

iter 140 | loss 3.6067

iter 150 | loss 4.1676

iter 160 | loss 3.2368

iter 170 | loss 3.3455

iter 180 | loss 3.3167

iter 190 | loss 3.7145

iter 200 | loss 3.2686

training time: 290.53788137435913

[200/300] Train loss: 3.67731, Valid loss: 3.46258, Elapsed_time: 643.82713

---

## Ground Truth | Prediction | Confidence Score & T/F

GUILLOT GOGUET | | 0.0064 False

JEAN COME | | 0.0063 False

---

validation time: 54.00260043144226

iter 210 | loss 3.1327

iter 220 | loss 3.8782

iter 230 | loss 3.3630

iter 240 | loss 3.4545

iter 250 | loss 3.1189

iter 260 | loss 3.2864

iter 270 | loss 3.8032

iter 280 | loss 3.5028

iter 290 | loss 3.0270

iter 300 | loss 2.9997

training time: 251.15985083580017

[300/300] Train loss: 3.48534, Valid loss: 3.45151, Elapsed_time: 948.98977

---

## Ground Truth | Prediction | Confidence Score & T/F

ELBAKKALI | MA | 0.0001 False

CLAIRE | MA | 0.0001 False

---

validation time: 65.48364853858948

### TESTING RESULTS 2:

super().**init**(
iter 0 | loss 48.6029

iter 10 | loss 4.5543

iter 20 | loss 4.7243

iter 30 | loss 4.0003

iter 40 | loss 4.5361

iter 50 | loss 3.5435

iter 60 | loss 5.2376

iter 70 | loss 3.5753

iter 80 | loss 4.1038

iter 90 | loss 4.2777

iter 100 | loss 3.5527

training time: 194.40260910987854

[100/600] Train loss: 4.62161, Valid loss: 3.69194,

---

## Ground Truth | Prediction | Confidence Score & T/F

GUILLOT GOGUET | | 0.0009 False

PAULINE | | 0.0028 False

---

validation time: 49.05067753791809

iter 110 | loss 5.2451

iter 120 | loss 4.1599

iter 130 | loss 4.2573

iter 140 | loss 3.7255

iter 150 | loss 3.4096

iter 160 | loss 3.4484

iter 170 | loss 3.3179

iter 180 | loss 3.5312

iter 190 | loss 3.6094

iter 200 | loss 3.2921

training time: 270.2250759601593

[200/600] Train loss: 3.65330, Valid loss: 3.32930, Elapsed_time: 513.68147

---

## Ground Truth | Prediction | Confidence Score & T/F

JEAN COME | | 0.0000 False

FRANSOISSISEPH | | 0.0000 False

---

validation time: 74.76693844795227

iter 210 | loss 4.6947

iter 220 | loss 3.5391

iter 230 | loss 3.7458

iter 240 | loss 3.4104

iter 250 | loss 3.2844

iter 260 | loss 3.4547

iter 270 | loss 3.0345

iter 280 | loss 4.9156

iter 290 | loss 3.4436

iter 300 | loss 3.4493

training time: 270.3188474178314

[300/600] Train loss: 3.53309, Valid loss: 3.22446, Elapsed_time: 858.76740

---

## Ground Truth | Prediction | Confidence Score & T/F

PAUL | L | 0.0013 False

Catherine | A | 0.0015 False

---

validation time: 60.673492431640625

iter 310 | loss 3.5023

iter 320 | loss 3.0998

iter 330 | loss 3.0922

iter 340 | loss 3.0576

iter 350 | loss 3.1524

iter 360 | loss 6.3221

iter 370 | loss 3.4671

iter 380 | loss 3.0475

iter 390 | loss 3.5319

iter 400 | loss 2.8710

training time: 289.7448694705963

[400/600] Train loss: 3.39331, Valid loss: 4.98558,

---

## Ground Truth | Prediction | Confidence Score & T/F

RELIS | M | 0.0004 False

JADE | M | 0.0000 False

---

validation time: 54.69872522354126

iter 410 | loss 3.4234

iter 420 | loss 3.1857

iter 430 | loss 3.4812

iter 440 | loss 3.2964

iter 450 | loss 3.5359

iter 460 | loss 3.1243

iter 470 | loss 2.7428

iter 480 | loss 3.2248

iter 490 | loss 3.1902

iter 500 | loss 3.6683

training time: 259.12396025657654

[500/600] Train loss: 3.39295, Valid loss: 3.27525,

---

## Ground Truth | Prediction | Confidence Score & T/F

RAPHAEL | B | 0.0003 False

BOURQUIN | | 0.0012 False

---

validation time: 53.25653696060181

iter 510 | loss 2.8886

iter 520 | loss 3.2058

iter 530 | loss 3.3698

iter 540 | loss 3.2743

iter 550 | loss 3.3068

iter 560 | loss 2.7712

iter 570 | loss 3.5144

iter 580 | loss 3.6514

iter 590 | loss 3.2486

iter 600 | loss 3.2205

training time: 242.94145488739014

[600/600] Train loss: 3.34676, Valid loss: 3.22076,

---

## Ground Truth | Prediction | Confidence Score & T/F

LILOU | D | 0.0002 False

ANNE | D | 0.0002 False

---

validation time: 56.336950063705444

end the training
