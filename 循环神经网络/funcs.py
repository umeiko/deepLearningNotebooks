"""
将自然语言处理的一些常用函数及类进行了打包
"""
import os
import collections
import random
import re
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Union, Callable
from tqdm.notebook import tqdm

forward_ = Callable[[torch.Tensor, torch.Tensor, list], tuple]
state_   = Callable[[int, int, str], torch.Tensor]
loss_    = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
param_   = Callable[[int, int, str], list]

class tqdmEnumerate():
    """兼容tqdm的enumerate模块"""
    def __init__(self, it, start:int = 0) -> None:
        self.enum = enumerate(it, start)
        self.it = it
    def __len__(self):
        return len(self.it)
    def __next__(self):
        return self.enum.__next__()
    def __iter__(self):
        return self.enum.__iter__()

def read_time_machine(path:str="./data_set/H_G_Well_time_machine.txt")->list:
    """从文件中读取'时光机器'数据集, 并按行隔开"""
    with open(path, "r") as f:
        lines = f.readlines()
    # 忽略标点符号和字母大写
    out = []
    for line in lines:
        # ^取非 +连续选取多个匹配
        line = re.sub('[^A-Za-z]+', ' ', line).strip().lower()
        if line:
            out.append(line)
    return out

def tokenize(lines:list, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

def read_data_nmt(data_path="./data_set/fra-eng/"):
    """载入“英语－法语”数据集"""
    with open(os.path.join(data_path, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

def tokenize_nmt(text:str, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for k, line in enumerate(text.split('\n')):
        if (num_examples is not None) and (k > num_examples):
            break
        pair = line.split('\t')
        if len(pair) == 2:
            source.append(pair[0].split(" "))
            target.append(pair[1].split(" "))
    return source, target

def preprocess_nmt(text:str):
    """文本预处理"""
    # 替换空白字符为普通空格，并转为全小写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '
    out = [" " + char if k>0 and no_space(char, text[k-1]) else char for k, char in enumerate(tqdm(text))]
    return "".join(out)

def truncate_pad(line:list[int], num_steps:int, padding_token:list[int]):
    """截断或填充文本序列
    
    输入参数
    ----------
    `line`     : 输入的句子
    `num_steps`: 需要调整到的长度
    `padding_token` : 长度不足时使用的填充词元
    """
    if len(line) > num_steps:
        return line[:num_steps]
    else:
        return line + [padding_token] * (num_steps - len(line))


class Vocab:
    """
    一个词元字典类的实现，会根据输入数据序列，统计出现词元的频率，并将词元安装词频进行排列.
    提供了词元的embedding方法，可以将词元编码为整数，或通过整数索引到某个词元。
    
    构造参数
    ----------
        `tokens`   : 包含词元的句子或其它形式的序列
        `min_freq` : 过滤掉出现频率低于该数量的词元
        `reserved_tokens` : 预设保留词元的序列
    """
    def __init__(self, tokens:list, min_freq=0, reserved_tokens:list=None) -> None:
        if tokens is not None:
            # 当第一个条件满足时，就不会跳到第二个判断，避免了空列表报错的情况。
            if len(tokens)!=0 and isinstance(tokens[0], list):
                tokens = [i for line in tokens for i in line]
        else:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter=collections.Counter(tokens)
        # 按出现词频从高到低排序
        self._token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        # 通过列表,利用序号访问词元。
        self.idx_to_token = ['<unk>'] + reserved_tokens # 未知词元<unk>的索引为0, 保留词元排在最前
        self.token_to_idx = {
            i: k
            for k, i in enumerate(self.idx_to_token) 
        }
        
        for token, freq in self._token_freqs:
            if freq < min_freq:  # 过滤掉出现频率低于要求的词
                break
            if token not in self.token_to_idx:  
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, input_tokens:Union[str, list]):
        """输入单字串或序列, 将其全部转化为序号编码"""
        if isinstance(input_tokens, str):
            return self.token_to_idx.get(input_tokens, 0)
        return [self.__getitem__(token) for token in input_tokens]
    
    def __repr__(self) -> str:
        show_items = 5 if len(self) > 5 else len(self)
        out = f"<Vocab with {len(self)} tokens: "
        for i in range(show_items):
            out += f'"{self.idx_to_token[i]}", '
        out += "...>"
        return out

    def to_tokens(self, input_keys):
        """输入单索引或序列, 将其全部转化为词元"""
        if isinstance(input_keys, int):
            return self.idx_to_token[input_keys] if input_keys < len(self) else self.idx_to_token[0]
        elif isinstance(input_keys, (list, tuple)):
            return [self.to_tokens(keys) for keys in input_keys]
        else:
            return self.idx_to_token[0]

def build_array_nmt(lines:list, vocab:Vocab, num_steps:int):
    """将文本序列转换为小批量

    在每个句子句尾添加`<eos>`标识。
    将句子截断或填充到指定长度。
    转化为张量输出。
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor(
        [truncate_pad(l, num_steps, vocab["<pad>"]) for l in lines]
        )
    valid_len = (array != vocab["<pad>"]).to(torch.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size:int, num_steps:int, num_examples=600):
    """返回翻译数据集迭代器及词表
    
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    reserved_tokens = ['<pad>', "<bos>", "<eos>"]
    src_vocab = Vocab(source, 2, reserved_tokens)
    tgt_vocab = Vocab(target, 2, reserved_tokens)
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    
    dataset = TensorDataset(*data_arrays)
    data_iter = DataLoader(dataset, batch_size, shuffle=True)
    return data_iter, src_vocab, tgt_vocab


def seq_data_iter_sequential(corpus:list, batch_size:int, num_steps:int, drop_last:bool=True):
    """ 实现顺序分区策略 """
    last_ = 1 if drop_last else 0
    # 舍去前端来产生偏移
    corpus = corpus[random.randint(0, num_steps-1):]
    # 索引数目
    num_samples = (len(corpus) - 1) // (num_steps * batch_size + last_)
    # 生成起始词的索引值
    init_indices = [i * (num_steps * batch_size) for i in range(num_samples+1)]
    # 打乱索引
    random.shuffle(init_indices)
    out_x, out_y = [], []
    for _, i in enumerate(init_indices):
        for items in range(batch_size):
            temp_x = corpus[i+items*num_steps : i+items*num_steps+num_steps]
            temp_y = corpus[i+items*num_steps+1:i+items*num_steps+num_steps+1]
            if len(temp_x) == len(temp_y) and len(temp_x) == num_steps:
                out_x.append(temp_x)
                out_y.append(temp_y)     
        if drop_last and len(out_x) < batch_size:
            out_x, out_y = [], []
            continue
        yield out_x, out_y
        out_x, out_y = [], []

def seq_data_iter_random(corpus:list, batch_size:int, num_steps:int, drop_last:bool=True):
    """
        实现随机采样策略
        参数：词组序列, 批量, 分区长度, 生成 序列 及 该序列的预测结果(后面紧跟的一个词元)
    """
    last_ = 1 if drop_last else 0
    # 舍去前端来产生偏移
    corpus = corpus[random.randint(0, num_steps-1):]
    # 索引数目
    num_samples = (len(corpus) - 1) // (num_steps + last_)
    # 生成起始词的索引值
    init_indices = [i * (num_steps) for i in range(num_samples+1)]
    # 打乱索引
    random.shuffle(init_indices)
    out_x:list = []
    out_y:list = []
    for k, i in enumerate(init_indices):
        temp_x = corpus[i:i+num_steps]
        temp_y = corpus[i+1:i+num_steps+1]
        if len(temp_x) == len(temp_y) and len(temp_x) == num_steps:
            out_x.append(temp_x)
            out_y.append(temp_y) 
        if (k+1) % batch_size == 0:
            yield out_x, out_y
            out_x, out_y = [], []
    if len(out_x) == batch_size:
        yield out_x, out_y
    elif out_x and not drop_last:
        yield out_x, out_y

class SeqDataLoaderTimeMachine:
    """加载时光机器序列数据用的迭代器"""
    def __init__(self, batch_size:int, num_steps:int, use_random_iter:bool,
                    max_tokens:int, token_type:str="word"):
        self.data_iter = seq_data_iter_random if use_random_iter else seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens, token_type)
        self.batch_size, self.num_steps = batch_size, num_steps
    
    def __iter__(self):
        return self.data_iter(self.corpus, self.batch_size, self.num_steps)
    
    def __repr__(self) -> str:
        show_items = 5 if len(self.corpus) > 5 else len(self.corpus)
        out = f"<TimeMachineLoader with {len(self.corpus)} corpus: "
        for i in range(show_items):
            out += f'"{self.vocab.to_tokens(self.corpus[i])}", '
        out += "...>"
        return out
    
    def __len__(self):
        return len(self.corpus)

def load_corpus_time_machine(max_tokens=-1, token_type="word"):
    """读取时光机器数据集并转化为词编码和解码词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, token_type)
    vocab = Vocab(tokens)
    # 将文本行展平放入一个段落中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def load_data_time_machine(batch_size:int, num_steps:int, 
                            use_random_iter:bool=False, 
                            max_tokens:int=10000,
                            token_type="word"):
    """返回时光机器数据集迭代器及其词表"""
    data_iter = SeqDataLoaderTimeMachine(
        batch_size, num_steps, use_random_iter, max_tokens, token_type)
    return data_iter, data_iter.vocab

class Timer():
    """简易计时器
    初始化后即开始计时
    `Timer.stop()`: 停止计时并返回计时时间
    """
    def __init__(self):
        self.start_time = time.time()
    def stop(self)-> float:
        return time.time() - self.start_time

class Accumulator():
    """参数计数器"""
    def __init__(self, item_count:int):
        self.lst = [ 0 for _ in range(item_count)]
    def add(self, *args):
        if len(args) > len(self.lst):
            raise Exception("The input item count should not exceed the defined item count.")
        for k, i in enumerate(args):
            self.lst[k] += i
    def __getitem__(self, idx):
        return self.lst[idx]

def grad_clipping(net:nn.Module, theta:float):
    """梯度裁剪"""
    params = [p for p in net.parameters() if p.requires_grad]
    # 计算 ||g||
    norm = torch.sqrt( sum(torch.sum(p.grad ** 2) for p in params) )
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm     

def rnn_predict(prefix:str, num_preds:int, net, vocab:Vocab,  device:str, add_blank=False):
    """通过序列模型, 续写一段文字"""
    prefix    = prefix.split() if add_blank else prefix
    state     = net.begin_state(batch_size=1, device=device)
    outputs   = [vocab[prefix[0]]]
    # 每次以输出序列的最后一个字符作为输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
    # 这个阶段作为预热阶段
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 这个阶段为预测阶段
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=-1).reshape(1)))
    blank = " " if add_blank else ""
    return ''.join([vocab.idx_to_token[i]+blank for i in outputs])

class RNN(nn.Module):
    """循环神经网络的自己实现"""
    def __init__(self, vocab_size:int, num_hiddens:int, device:str, 
                    get_params:param_, init_state:state_, 
                    forward_fn:forward_) -> None:
        super().__init__()
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    
    def forward(self, x:torch.Tensor, state:torch.Tensor):
        x = F.one_hot(x.T, self.vocab_size).to(torch.float32)
        return self.forward_fn(x, state, self.params)
        
    def begin_state(self, batch_size:int, device:str):
        return self.init_state(batch_size, self.num_hiddens, device)
    
    def parameters(self, recurse: bool = True):
        for param in self.params:
            yield param

    def __repr__(self) -> str:
        return f"<RNN_Module with {self.num_hiddens} hiddens>"

class RNN_(nn.Module):
    """利用高级API的实现"""
    def __init__(self, layer:nn.RNN, device="cuda:1", batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.device = device
        self.layer = layer
        self.num_layers = layer.num_layers
        self.num_hiddens = layer.hidden_size
        # 处理模型为双向模型时的问题
        if not self.layer.bidirectional:
            self.out_layer = nn.Linear(layer.hidden_size, layer.input_size, device=device)
        else:
            self.out_layer = nn.Linear(layer.hidden_size*2, layer.input_size, device=device)
    
    def forward(self, x:torch.Tensor, state:torch.Tensor):
        x = x.long().to(self.device)
        if self.batch_first:
            x = F.one_hot(x, self.layer.input_size).to(torch.float)
        else:
            x = F.one_hot(x.T, self.layer.input_size).to(torch.float)
        y, state = self.layer(x, state)
        # y:[t,b,embed]  ->  [t*b, embed]
        out = self.out_layer(y.reshape((-1, y.shape[-1])))
        return out, state
    
    def begin_state(self, batch_size:int, device:str):
        num_layers = self.num_layers if not self.layer.bidirectional else (self.num_layers * 2)
        if isinstance(self.layer, nn.LSTM):
            return (torch.zeros((num_layers, batch_size, self.num_hiddens), device=device),
            torch.zeros((num_layers, batch_size, self.num_hiddens), device=device))
        else:
            return torch.zeros((num_layers, batch_size, self.layer.hidden_size), 
                            dtype=torch.float, device=device)
    

def train_rnn_one_epoch(net:RNN, train_iter:SeqDataLoaderTimeMachine, 
                        loss:loss_, opt:torch.optim.Optimizer,
                        device:str, use_random_iter:bool, batch_first=False):
    """返回结果:困惑度, 训练速度 (词元/秒) """
    state = None
    loss_count, num_tokens = 0, 0
    timer = Timer()
    for x, y in train_iter:
        if not batch_first:
            # 转置后展平，标签数据按照时序排列
            y = torch.tensor(y).T.reshape(-1)
        else:
            # 【这里y不转置，直接按照批次排序】
            y = torch.tensor(y).reshape(-1)
        x, y = torch.tensor(x).to(device), y.to(device)
        # 需要初始化隐变量的情况
        if state is None or use_random_iter:
            # 如果使用随机抽样方法，则每个minibatch都重新初始化state
            state = net.begin_state(x.shape[0], device)
        else:
            state = state_.detach()
        y_hat, state = net(x, state)
        l = loss(y_hat, y.to(torch.long)).mean()
        # 更新梯度
        opt.zero_grad()
        l.backward()
        grad_clipping(net, 1.)
        opt.step()
        # .numel() 返回张量中的所有参数数量
        loss_count += l * y.numel()
        num_tokens += y.numel()
    with torch.no_grad():
        return torch.exp(loss_count / num_tokens), num_tokens / timer.stop()

def train_rnn(net:RNN, num_epochs, train_iter, opt:torch.optim.Optimizer, device, tqdm=None, batch_first=False):
    """训练循环神经网络"""
    loss = nn.CrossEntropyLoss()
    net.train()
    ppl_ = []

    if tqdm is not None:
        iter_ = tqdm(range(num_epochs))
    else:
        iter_ = range(num_epochs)

    for epo in iter_:
        ppl, speed = train_rnn_one_epoch(
            net, train_iter, loss, opt, device, True, batch_first)
        if (epo + 1) % 10 == 0:
            ppl_.append(float(ppl))

    print(f'困惑度[{ppl:.2f}], 速度[{speed:.1f} 词元/秒], 设备[{str(device)}]')
    return ppl_


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, X, state) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder:Encoder, decoder:Decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)