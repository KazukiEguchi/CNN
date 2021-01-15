import numpy as np
from collections import OrderedDict
from layers import Convolution, MaxPooling, ReLU, Affine, SoftmaxWithLoss,BatchNormalization

class Three_ConvNet:
    """
        input_size : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param : dict, 畳み込みの条件
        pool_param : dict, プーリングの条件
        hidden_size : int, 隠れ層のノード数
        output_size : int, 出力層のノード数
        weight_init_std : float, 重みWを初期化する際に用いる標準偏差
        filter_num_list : filterの数のリスト
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_size':3, 'pad':1, 'stride':1},
                 pool_param={'pool_size':2, 'pad':0, 'stride':2},filter_num_list=[32,64,64],
                 hidden_size=100, output_size=15, weight_init_std=0.01):
        
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        pool_size = pool_param['pool_size']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']
        self.filter_num_list = filter_num_list
        
        # 重みの初期化
        self.params = {}
        std = weight_init_std
        cv_input_size = input_dim[1]
        cv_list = filter_num_list
        input_oku = input_dim[0]
        for idx in range(1, len(cv_list)+1):
            filter_num  = cv_list[idx - 1]
            conv_output_size = (cv_input_size + 2*filter_pad - filter_size) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
            pool_output_size = (conv_output_size + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
            
            self.params['W'+ str(idx)] = std * np.random.randn(filter_num,input_oku, filter_size, filter_size) # 畳み込みフィルターの重み
            self.params['b'+ str(idx)] = np.zeros(filter_num) #畳み込みフィルターのバイアス
            self.params['gamma'+ str(idx)] = np.ones(filter_num)
            self.params['beta'+ str(idx)] = np.zeros(filter_num)
            
            cv_input_size = pool_output_size
            input_oku = filter_num
        
        pool_output_pixel = filter_num * pool_output_size * pool_output_size # プーリング後のピクセル総数
        self.params['W_hidden'] = std *  np.random.randn(pool_output_pixel, hidden_size)
        self.params['b_hidden'] = np.zeros(hidden_size)
        self.params['W_last'] = std *  np.random.randn(hidden_size, output_size)
        self.params['b_last'] = np.zeros(output_size)
        self.params['gamma_last'] = np.ones(hidden_size)
        self.params['beta_last'] = np.zeros(hidden_size)
        
        # レイヤの生成
        self.layers = OrderedDict()
        for idx in range(1, len(cv_list)+1):
            self.layers['Conv'+ str(idx)] = Convolution(self.params['W'+ str(idx)], self.params['b'+ str(idx)],conv_param['stride'], conv_param['pad'])
            self.layers['BatchNorm'+ str(idx)] = BatchNormalization(self.params['gamma'+ str(idx)], self.params['beta'+ str(idx)])
            self.layers['ReLU'+ str(idx)] = ReLU()
            self.layers['Pool'+ str(idx)] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)
        self.layers['Affine_hidden'] = Affine(self.params['W_hidden'], self.params['b_hidden'])
        self.layers['BatchNorm_last'] = BatchNormalization(self.params['gamma_last'], self.params['beta_last'])
        self.layers['ReLU_last'] = ReLU()
        self.layers['Affine_last'] = Affine(self.params['W_last'], self.params['b_last'])

        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, x,train_flg=False):
        for key, layer in self.layers.items():
            if "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        
        return x

    def loss(self, x, t,train_flg=False):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x,train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx,train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t,train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, len(self.filter_num_list)+1):
            grads['W'+ str(idx)], grads['b'+ str(idx)] = self.layers['Conv'+ str(idx)].dW, self.layers['Conv'+ str(idx)].db
            grads['gamma'+ str(idx)] = self.layers['BatchNorm'+ str(idx)].dgamma
            grads['beta'+ str(idx)] = self.layers['BatchNorm'+ str(idx)].dbeta
        
        grads['W_hidden'], grads['b_hidden'] = self.layers['Affine_hidden'].dW, self.layers['Affine_hidden'].db
        grads['gamma_last'] = self.layers['BatchNorm_last'].dgamma
        grads['beta_last'] = self.layers['BatchNorm_last'].dbeta
        grads['W_last'], grads['b_last'] = self.layers['Affine_last'].dW, self.layers['Affine_last'].db

        return grads