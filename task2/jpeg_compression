"""
This code uses RGB2YUV, DCT, quantization, zigzag, running length coding to compress the image (8x8 block size, JPEG compression).
Huffman coding is not included in this demo.
You can find a more comprehensive JPEG compression at: https://github.com/ghallak/jpeg-python
... are the positions you need to fill.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2


class KJPEG:
    def __init__(self):
        # generate the dct matrix
        self.__dctA = np.zeros(shape=(8, 8))
        for i in range(8):
            c = 0
            if i == 0:
                c = np.sqrt(1 / 8)
            else:
                c = np.sqrt(2 / 8)
            for j in range(8):
                self.__dctA[i, j] = c * np.cos(np.pi * i * (2 * j + 1) / (2 * 8))

        #####################################################################################################################
        # luminance quantization table, change it to control the compression ratio
        # (note: the low-frequency is more important for image content.)
        self.__lq = np.array([
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99,
        ])
        # self.__lq = np.array([
        #     32, 22, 20, 32, 48, 80, 102, 122,
        #     24, 24, 28, 38, 52, 116, 120, 110,
        #     28, 26, 32, 48, 80, 114, 138, 112,
        #     28, 34, 44, 58, 102, 174, 160, 124,
        #     36, 44, 74, 112, 136, 218, 206, 154,
        #     48, 70, 110, 128, 162, 208, 226, 184,
        #     98, 128, 156, 174, 206, 242, 240, 202,
        #     144, 184, 190, 196, 224, 200, 206, 198,
        # ])
        # chrom quantization table, change it to control the compression ratio
        # (note: the low-frequency is more important for image content.)
        self.__cq = np.array([
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
        ])
        # self.__cq = np.array([
        #     34, 36, 48, 94, 198, 198, 198, 198,
        #     36, 42, 52, 132, 198, 198, 198, 198,
        #     48, 52, 112, 198, 198, 198, 198, 198,
        #     94, 132, 198, 198, 198, 198, 198, 198,
        #     198, 198, 198, 198, 198, 198, 198, 198,
        #     198, 198, 198, 198, 198, 198, 198, 198,
        #     198, 198, 198, 198, 198, 198, 198, 198,
        #     198, 198, 198, 198, 198, 198, 198, 198,
        # ])
        #####################################################################################################################
        self.__lt = 0  # tag, represent luminance
        self.__ct = 1  # tag, represent chrom
        # the pixel index used to construct a new zigzag-scanned array.
        # for convinent, indexs are pre-computed and given below.
        self.__zigzag_index = np.array([
            0, 1, 8, 16, 9, 2, 3, 10,
            17, 24, 32, 25, 18, 11, 4, 5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63
        ])
        self.__izigzag_index = np.array([
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 41, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        ])

    # RGB to YUV
    def __Rgb2Yuv(self, r, g, b):
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
        v = 0.5 * r - 0.419 * g - 0.081 * b + 128
        return y, u, v

    # image padding, cannot be divided by 8.
    def __Fill(self, matrix):
        fh, fw = 0, 0
        if self.height % 8 != 0:
            fh = 8 - self.height % 8
        if self.width % 8 != 0:
            fw = 8 - self.width % 8
        matrix_padded = np.pad(matrix, ((0, fh), (0, fw)), 'constant', constant_values=(0, 0))
        return matrix_padded

    def __Encode(self, matrix, tag):
        """
        :param matrix:  Y or U or V
        :param tag: luminance (0) or chrom (1)
        :return:
        """
        matrix = self.__Fill(matrix)
        height, width = matrix.shape
        shape = (height // 8, width // 8, 8, 8)
        blocks = np.zeros(shape)
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                try:
                    block_index += 1
                except NameError:
                    block_index = 0
                block = matrix[i:i+8, j:j+8]
                blocks[i//8, j//8, :, :] = block
        res = []
        for i in range(height // 8):
            for j in range(width // 8):
                res.append(self.__Quantize(self.__Dct(blocks[i, j]).reshape(64), tag))
        return res

    def __Dct(self, block):
        block_DCT = np.dot(self.__dctA, block)
        block_DCT = np.dot(block_DCT, np.transpose(self.__dctA))
        return block_DCT

    #####################################################################################################################
    # please implement quantization
    def __Quantize(self, block, tag):
        if tag == self.__lt:
            block_quantize = np.round(block / self.__lq).reshape(8, 8)
        elif tag == self.__ct:
            block_quantize = np.round(block / self.__cq).reshape(8, 8)
        return block_quantize
    #####################################################################################################################

    #####################################################################################################################
    def __ZigZag(self, blocks):
        num_blocks = len(blocks)
        blocks_zigzag = []
        for i in range(num_blocks):
            block = blocks[i]
            block_zigzag = np.array([block.ravel()[self.__zigzag_index].reshape(8, 8)])
            blocks_zigzag.append(block_zigzag)
        blocks_zigzag = np.array(blocks_zigzag).flatten()
        return blocks_zigzag.tolist()
    #####################################################################################################################

    def __Rle(self, blist):
        rlist = []
        cnt = 0
        for i in range(len(blist)):
            if blist[i] != 0:
                rlist.append(cnt)
                rlist.append(int(blist[i]))
                cnt = 0
            elif cnt == 15:
                rlist.append(cnt)
                rlist.append(int(blist[i]))
                cnt = 0
            else:
                cnt += 1
        if cnt != 0:
            rlist.append(cnt - 1)
            rlist.append(0)
        return rlist

    def Compress(self, filename):
        image = cv2.imread(filename)[:, :, ::-1]
        self.width, self.height, self.channel = image.shape
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        y, u, v = self.__Rgb2Yuv(r, g, b)
        y_blocks = self.__Encode(y, self.__lt)
        u_blocks = self.__Encode(u, self.__ct)
        v_blocks = self.__Encode(v, self.__ct)
        y_code = self.__Rle(self.__ZigZag(y_blocks))
        u_code = self.__Rle(self.__ZigZag(u_blocks))
        v_code = self.__Rle(self.__ZigZag(v_blocks))
        tfile = os.path.splitext(filename)[0] + ".gpj"
        if os.path.exists(tfile):
            os.remove(tfile)
        with open(tfile, 'wb') as o:
            o.write(self.height.to_bytes(2, byteorder='big'))
            o.flush()
            o.write(self.width.to_bytes(2, byteorder='big'))
            o.flush()
            o.write((len(y_code)).to_bytes(4, byteorder='big'))
            o.flush()
            o.write((len(u_code)).to_bytes(4, byteorder='big'))
            o.flush()
            o.write((len(v_code)).to_bytes(4, byteorder='big'))
            o.flush()
        self.__Write2File(tfile, y_code, u_code, v_code)

    def __Write2File(self, filename, y_code, u_code, v_code):
        with open(filename, "ab+") as o:
            buff = 0
            bcnt = 0
            data = y_code + u_code + v_code
            for i in range(len(data)):
                if i % 2 == 0:
                    td = data[i]
                    for ti in range(4):
                        buff = (buff << 1) | ((td & 0x08) >> 3)
                        td <<= 1
                        bcnt += 1
                        if bcnt == 8:
                            o.write(buff.to_bytes(1, byteorder='big'))
                            o.flush()
                            buff = 0
                            bcnt = 0
                else:
                    td = data[i]
                    vtl, vts = self.__VLI(td)
                    for ti in range(4):
                        buff = (buff << 1) | ((vtl & 0x08) >> 3)
                        vtl <<= 1
                        bcnt += 1
                        if bcnt == 8:
                            o.write(buff.to_bytes(1, byteorder='big'))
                            o.flush()
                            buff = 0
                            bcnt = 0
                    for ts in vts:
                        buff <<= 1
                        if ts == '1':
                            buff |= 1
                        bcnt += 1
                        if bcnt == 8:
                            o.write(buff.to_bytes(1, byteorder='big'))
                            o.flush()
                            buff = 0
                            bcnt = 0
            if bcnt != 0:
                buff <<= (8 - bcnt)
                o.write(buff.to_bytes(1, byteorder='big'))
                o.flush()
                buff = 0
                bcnt = 0

    def __IDct(self, block):
        res = np.dot(np.transpose(self.__dctA), block)
        res = np.dot(res, self.__dctA)
        return res

    def __IQuantize(self, block, tag):
        res = block
        if tag == self.__lt:
            res *= self.__lq
        elif tag == self.__ct:
            res *= self.__cq
        return res

    def __IFill(self, matrix):
        matrix = matrix[:self.height, :self.width]
        return matrix

    def __Decode(self, blocks, tag):
        tlist = []
        for b in blocks:
            b = np.array(b)
            tlist.append(self.__IDct(self.__IQuantize(b, tag).reshape(8 ,8)))
        height_fill, width_fill = self.height, self.width
        if height_fill % 8 != 0:
            height_fill += 8 - height_fill % 8
        if width_fill % 8 != 0:
            width_fill += 8 - width_fill % 8
        rlist = []
        for hi in range(height_fill // 8):
            start = hi * width_fill // 8
            rlist.append(np.hstack(tuple(tlist[start: start + (width_fill // 8)])))
        matrix = np.vstack(tuple(rlist))
        res = self.__IFill(matrix)
        return res

    def __ReadFile(self, filename):
        with open(filename, "rb") as o:
            tb = o.read(2)
            self.height = int.from_bytes(tb, byteorder='big')
            tb = o.read(2)
            self.width = int.from_bytes(tb, byteorder='big')
            tb = o.read(4)
            ylen = int.from_bytes(tb, byteorder='big')
            tb = o.read(4)
            ulen = int.from_bytes(tb, byteorder='big')
            tb = o.read(4)
            vlen = int.from_bytes(tb, byteorder='big')
            buff = 0
            bcnt = 0
            rlist = []
            itag = 0
            icnt = 0
            vtl, tb, tvtl = None, None, None
            while len(rlist) < ylen + ulen + vlen:
                if bcnt == 0:
                    tb = o.read(1)
                    if not tb:
                        break
                    tb = int.from_bytes(tb, byteorder='big')
                    bcnt = 8
                if itag == 0:
                    buff = (buff << 1) | ((tb & 0x80) >> 7)
                    tb <<= 1
                    bcnt -= 1
                    icnt += 1
                    if icnt == 4:
                        rlist.append(buff & 0x0F)
                    elif icnt == 8:
                        vtl = buff & 0x0F
                        tvtl = vtl
                        itag = 1
                        buff = 0
                else:
                    buff = (buff << 1) | ((tb & 0x80) >> 7)
                    tb <<= 1
                    bcnt -= 1
                    tvtl -= 1
                    if tvtl == 0 or tvtl == -1:
                        rlist.append(self.__IVLI(vtl, bin(buff)[2:].rjust(vtl, '0')))
                        itag = 0
                        icnt = 0
        y_dcode = rlist[:ylen]
        u_dcode = rlist[ylen:ylen+ulen]
        v_dcode = rlist[ylen+ulen:ylen+ulen+vlen]
        return y_dcode, u_dcode, v_dcode

    def __IZigZag(self, dcode):
        dcode = np.array(dcode).reshape((len(dcode) // 64, 64))
        tz = np.zeros(dcode.shape)
        for i in range(len(self.__izigzag_index)):
            tz[:, i] = dcode[:, self.__izigzag_index[i]]
        rlist = tz.tolist()
        return rlist

    def __IRle(self, dcode):
        rlist = []
        for i in range(len(dcode)):
            if i % 2 == 0:
                rlist += [0] * dcode[i]
            else:
                rlist.append(dcode[i])
        return rlist

    def Decompress(self, filename):
        y_dcode, u_dcode, v_dcode = self.__ReadFile(filename)
        y_blocks = self.__IZigZag(self.__IRle(y_dcode))
        u_blocks = self.__IZigZag(self.__IRle(u_dcode))
        v_blocks = self.__IZigZag(self.__IRle(v_dcode))
        y = self.__Decode(y_blocks, self.__lt)
        u = self.__Decode(u_blocks, self.__ct)
        v = self.__Decode(v_blocks, self.__ct)
        r = (y + 1.402 * (v - 128))
        g = (y - 0.34414 * (u - 128) - 0.71414 * (v - 128))
        b = (y + 1.772 * (u - 128))
        r = Image.fromarray(r).convert('L')
        g = Image.fromarray(g).convert('L')
        b = Image.fromarray(b).convert('L')
        image = Image.merge("RGB", (r, g, b))
        image.save("./result.bmp", "bmp")
        plt.imshow(image)
        plt.title("Recovered Image")
        plt.axis('off')
        plt.show()

    def __VLI(self, n):
        ts, tl = 0, 0
        if n > 0:
            ts = bin(n)[2:]
            tl = len(ts)
        elif n < 0:
            tn = (-n) ^ 0xFFFF
            tl = len(bin(-n)[2:])
            ts = bin(tn)[-tl:]
        else:
            tl = 0
            ts = '0'
        return (tl, ts)

    def __IVLI(self, tl, ts):
        if tl != 0:
            n = int(ts, 2)
            if ts[0] == '0':
                n = n ^ 0xFFFF
                n = int(bin(n)[-tl:], 2)
                n = -n
        else:
            n = 0
        return n


if __name__ == '__main__':
    kjpeg = KJPEG()
    kjpeg.Compress("./1.bmp")
    kjpeg.Decompress("./1.gpj")
