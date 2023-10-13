from socket import socket, SOCK_STREAM, \
    AF_INET, SOCK_DGRAM
from typing import Union

import numpy as np
from base.embedded.constant import *


class WifiConfig:
    def __init__(self):
        try:
            s = socket(AF_INET, SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            print("ip:", ip)
        finally:
            s.close()

        tmp = ip.split('.')

        carip = '.'.join(tmp[:-1] + ['', ]) + str(1)
        self.server_adder = (carip, 60000)
        # log.info(self.server_adder)
        print(self.server_adder)

        # AF_INET表示IPV4地址，SOCK_STREAM表示走TCP
        self.client = socket(AF_INET, SOCK_STREAM)

        # 建立连接
        self.client.connect(self.server_adder)

    def _checkData(self, pack_length=FRAME_LEN):
        while True:
            dat_buf = self.client.recv(pack_length)
            if len(dat_buf) == FRAME_LEN and dat_buf[0] == FRAME_HEAD:
                break
            else:
                print("skip error package!!")
                # log.error("skip error package!!")
        return dat_buf

    def datRead(self):
        wifi_msg = self._checkData()
        return wifi_msg

    def send(self, send_dat: Union[bytes, np.ndarray]):
        self.client.send(send_dat)

if __name__ == '__main__':
    dat = np.zeros((4,), np.uint8)
    dat[0] = 0x41
    dat[1] = 0x50
    wifi = WifiConfig()
    print(wifi.client)
    # wifi.send('AT'.encode('ascii'))
    wifi.send(dat)
    print('-' * 20)
    read = wifi.datRead()
    print(read)
    try:
        print(read.decode('ascii'))
    except UnicodeDecodeError:
        print('解码失败')
