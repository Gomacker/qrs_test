#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""数据帧长度"""
FRAME_LEN = 5

"""通信协议"""
FRAME_HEAD = 0x55    # 包头
FRAME_TAIL = 0xBB    # 包尾

''''帧头第二位'''
MAIN_CAR = 0xCC
