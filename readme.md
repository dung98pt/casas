# Cài đặt môi trường
pip install -r requirement.txt
# Tiền xử lý dữ liệu
python pre_process.py --n 2tt  --w 150  --d true --i -1
# Đào tạo mô hình
python train/train.py --n 2tt --w 50 --m LSTM
# Đánh giá mô hình 
python evaluate/evaluate_cate2.py --m LSTM_Embedded_2tt_150_BEST_.h5 --n 2tt --w 150 --c test