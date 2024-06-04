安裝環境：
    1. 安裝 [Visual Studio Code](https://code.visualstudio.com/)

    2. 安裝 [Anaconda](https://www.anaconda.com/download)

    3. 開啟Anaconda Prompt (anaconda3)

    4. 進入D槽，輸入指令 ˋD:ˋ

    5. 進到放此README.md的AICUP資料夾中，輸入指令如 ˋcd D:\AICUPˋ

    6. 創建虛擬環境，輸入指令 ˋconda create -n AICUP_nav_gen python=3.10ˋ

    7. 進入虛擬環境(AICUP_nav_gen)，輸入指令 ˋconda activate AICUP_nav_genˋ

    8. 安裝套件，輸入指令ˋpip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116ˋ

    9. 安裝requirements中的套件，輸入指令ˋpip install -r requirements.txtˋ

    10. 切換路徑，輸入指令 ˋcd D:\AICUP\Real-ESRGAN-masterˋ

    11. 輸入指令 ˋpython D:\AICUP\Real-ESRGAN-master\setup.py developˋ


圖片轉換:
    1. 雲端硬碟下載訓練圖片集 35_Competition 2_Training dataset_V3.zip 解壓縮放到 D:\AICUP\AI_cup_demo_code\pytorch-CycleGAN-and-pix2pix-master\datasets\

    2. 雲端硬碟下載公開測試圖片集 35_Competition 2_public testing dataset.zip 解壓縮放到 D:\AICUP\AI_cup_demo_code\pytorch-CycleGAN-and-pix2pix-master\datasets\

    3. 雲端硬碟下載不公開測試圖片集 35_Competition 2_Private Test Dataset.zip 解壓縮放到 D:\AICUP\AI_cup_demo_code\pytorch-CycleGAN-and-pix2pix-master\datasets\

    4. 開啟Visual Studio Code

    5. 檔案 -> 開啟資料夾 -> AICUP資料夾

    6. 使用 資料處理.ipynb，進行 "訓練圖片轉換" 和 "測試圖片轉換" ，路徑若不同需修改

    7. 使用Real-ESRGAN對河流的訓練圖片進行增強，檢查 D:\AICUP\Real-ESRGAN-master\inputs 中是否有增強前的河流訓練圖片

    8. 使用此連結 https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth 下載Real-ESRGAN的權重，並放到 D:\AICUP\Real-ESRGAN-master\weights\

    9. 進行Real-ESRGAN的轉換，在Visual Studio Code的終端(進虛擬環境)，輸入指令 ˋpython D:\AICUP\Real-ESRGAN-master\inference_realesrgan.py -i D:\AICUP\Real-ESRGAN-master\inputs -o D:\AICUP\Real-ESRGAN-master\outputs -n RealESRGAN_x4plus -s 1 --suffix "_"ˋ

    10. 使用 資料處理.ipynb，進行 "將Real-ESRGAN的outputs中的圖片後綴移除並移到pix2pix的datasets\train_river_Real_ESRGAN\train資料夾中" ，會將處理完的圖片放到D:\AICUP\AI_cup_demo_code\pytorch-CycleGAN-and-pix2pix-master\datasets\train_river_Real_ESRGAN\train 內

    11. 手動將 D:\AICUP\Real-ESRGAN-master\inputs 內的圖片刪除，將 D:\AICUP\AI_cup_demo_code\pytorch-CycleGAN-and-pix2pix-master\datasets\pub_and_pri_test_river\test 內的所有圖片複製到 D:\AICUP\Real-ESRGAN-master\inputs

    12. 進行Real-ESRGAN的轉換，在Visual Studio Code的終端(進虛擬環境)，輸入指令 ˋpython D:\AICUP\Real-ESRGAN-master\inference_realesrgan.py -i D:\AICUP\Real-ESRGAN-master\inputs -o D:\AICUP\Real-ESRGAN-master\outputs -n RealESRGAN_x4plus -s 1 --suffix "_"ˋ

    13. 使用 資料處理.ipynb，進行 "將Real-ESRGAN的outputs中的圖片後綴移除並移到pix2pix的datasets\pub_and_pri_test_river_Real_ESRGAN\test資料夾中" ，會將處理完的圖片放到D:\AICUP\AI_cup_demo_code\pytorch-CycleGAN-and-pix2pix-master\datasets\pub_and_pri_test_river_Real_ESRGAN\test 內


訓練方式:
    1. 使用 pix2pix_3(PC).ipynb，進行 "Road Training" ，在Visual Studio Code的終端(進虛擬環境)，輸入指令ˋpython D:/AICUP/AI_cup_demo_code/pytorch-CycleGAN-and-pix2pix-master/train.py --dataroot D:/AICUP/AI_cup_demo_code/pytorch-CycleGAN-and-pix2pix-master/datasets/train_road --name ROAD_pix2pix_road_g2_17 --model pix2pix --direction AtoB --netG unet_512 --netD pixel --save_epoch_freq 10 --preprocess noneˋ ，訓練完權重會放到D:\AICUP\checkpoints\ROAD_pix2pix_road_g2_17

    2. 使用 pix2pix_3(PC).ipynb，進行 "River Training" ，在Visual Studio Code的終端(進虛擬環境)，輸入指令ˋpython D:/AICUP/AI_cup_demo_code/pytorch-CycleGAN-and-pix2pix-master/train.py --dataroot D:/AICUP/AI_cup_demo_code/pytorch-CycleGAN-and-pix2pix-master/datasets/train_river_Real_ESRGAN --name ROAD_pix2pix_river_g2_18 --model pix2pix --direction AtoB --netG unet_512 --netD pixel --save_epoch_freq 10 --preprocess noneˋ ，訓練完權重會放到D:\AICUP\checkpoints\ROAD_pix2pix_river_g2_18


上傳方式:
    1. 使用 pix2pix_3(PC).ipynb，進行 "Road Testing" ，在Visual Studio Code的終端(進虛擬環境)，輸入指令ˋpython D:/AICUP/AI_cup_demo_code/pytorch-CycleGAN-and-pix2pix-master/test.py --dataroot D:/AICUP/AI_cup_demo_code/pytorch-CycleGAN-and-pix2pix-master/datasets/pub_and_pri_test_road --name ROAD_pix2pix_road17 --model pix2pix --checkpoints_dir D:/AICUP/checkpoints/ROAD_pix2pix_road_g2_17 --results_dir D:/AICUP/AI_cup_demo_code/results/ROAD_pix2pix_road17 --direction AtoB --netG unet_512 --netD pixel --preprocess noneˋ
    會先報錯一次，再手動將權重 latest_net_G.pth 移動到 D:\AICUP\checkpoints\ROAD_pix2pix_road_g2_17\ROAD_pix2pix_road17 內， 測試完會放到 D:\AICUP\AI_cup_demo_code\results\ROAD_pix2pix_road17\ROAD_pix2pix_road17\test_latest\images

    (訓練完的權重放在雲端硬碟的道路河流權重中，如需使用，放到 checkpoints\ROAD_pix2pix_river_g2_18\ROAD_pix2pix_river18 中)

    2. 使用 pix2pix_3(PC).ipynb，進行 "River Testing" ，在Visual Studio Code的終端(進虛擬環境)，輸入指令ˋpython D:/AICUP/AI_cup_demo_code/pytorch-CycleGAN-and-pix2pix-master/test.py --dataroot D:/AICUP/AI_cup_demo_code/pytorch-CycleGAN-and-pix2pix-master/datasets/pub_and_pri_test_river_Real_ESRGAN --name ROAD_pix2pix_river18 --model pix2pix --checkpoints_dir D:/AICUP/checkpoints/ROAD_pix2pix_river_g2_18 --results_dir D:/AICUP/AI_cup_demo_code/results/ROAD_pix2pix_river18 --direction AtoB --netG unet_512 --netD pixel --preprocess noneˋ
    會先報錯一次，再手動將權重 latest_net_G.pth 移動到 D:\AICUP\checkpoints\ROAD_pix2pix_river_g2_18\ROAD_pix2pix_river18 內，測試完會放到 D:\AICUP\AI_cup_demo_code\results\ROAD_pix2pix_river18\ROAD_pix2pix_river18\test_latest\images

    (訓練完的權重放在雲端硬碟的道路權重中，如需使用，放到 checkpoints\ROAD_pix2pix_road_g2_17\ROAD_pix2pix_road17 中)

    3. 使用 資料處理.ipynb， 進行 "提取出生圖片並改名且resize放到upload資料夾中"

    4. 手動對 D:\AICUP\AI_cup_demo_code\results 中的 upload資料夾 壓縮、改名，上傳 
