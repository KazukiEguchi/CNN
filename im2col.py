import numpy as np

def im2col(input_data,filter_h,filter_w,stride = 1,pad = 0, constant_values= 0):
    """
    input_date : 入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド数
    pad : パディングサイズ
    pad_value : パディング処理で埋める値
    """
    
    N,C,H,W = input_data.shape
    
    #出力データの形状
    out_h = (H + 2*pad - filter_h)//stride + 1 
    out_w = (W + 2*pad - filter_w)//stride + 1 

    # パディング処理
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)],'constant', constant_values=constant_values) 
    
    # 初期化
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) 

    # colに入力データ代入
    for y in range(filter_h):
        y_max = y + stride*out_h
        
        for x in range(filter_w):
            x_max = x + stride*out_w
            
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1) # 軸を入れ替えて、2次元配列(行列)に変換する
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col : 2次元配列
    input_shape : 戻すデータの形状
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド数
    pad : パディングサイズ
    return : (データ数, チャンネル数, 高さ, 幅)の4次元配列. 画像データの形式を想定している
    -------
    """
    
    # 入力画像(元画像)のデータ数, チャンネル数, 高さ, 幅を取得する
    N, C, H, W = input_shape
    
    # 出力の形状を計算する
    out_h = (H + 2*pad - filter_h)//stride + 1 
    out_w = (W + 2*pad - filter_w)//stride + 1 
    
    # colを6次元配列に
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    #初期化
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))  # pad分を大きくとる. stride分も大きくとる
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            
            # colから値を取り出し、imgに入れる
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            
    return img[:, :, pad:H + pad, pad:W + pad] # pad分は除いておく(pad分を除いて真ん中だけを取り出す)