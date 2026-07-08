import math
import serial

class SerialAutocolor:
    def __init__(
            self,
            PI=3.1415,
            LLGMN_MAX_IN_CH=16,  # LLGMN入力層の最大ユニット数(1+ EMG_MAX_CH*(3+EMG_MAX_CH)/2 + 1)
            MAX_COMP_NO=3,  # LLGMNの最大コンポーネント数
            EMG_MAX_CH=4,  # 最大次元数
            MAX_MOTION_NO=3,  # 最大動作数
            MAX_SAMPLE_NO=100,  # 最大サンプル数
            # CLASS_DT_SIZE = 8080,  #識別するサンプル数（最大サンプル数 × 最大動作数）リアルタイムでは使わない
    ):
        super().__init__()

        self.count_color = 0  # 色の指定用
        self.change_color = 0  # 回外を数える用

        # グローバル変数
        self.comp_no = MAX_COMP_NO
        self.emg_ch = EMG_MAX_CH
        self.motion_no = MAX_MOTION_NO
        self.sample_no = MAX_SAMPLE_NO
        # self.class_dt_size = CLASS_DT_SIZE  # リアルタイムでは使わない
        self.dummy = [
            [0.0 for _ in range(MAX_SAMPLE_NO * MAX_MOTION_NO + 1)]
            for _ in range(EMG_MAX_CH + 1)
        ]
        # dummy0 = [[0.0 for _ in range(MAX_SAMPLE_NO * MAX_MOTION_NO + 1)] for _ in range(EMG_MAX_CH + 1)]
        self.dummy0 = []
        self.dummy_sum = [0.0] * (MAX_SAMPLE_NO * MAX_MOTION_NO + 1)
        self.x = [
            [0.0 for _ in range(MAX_SAMPLE_NO * MAX_MOTION_NO + 1)]
            for _ in range(1 + EMG_MAX_CH * (EMG_MAX_CH + 3) // 2 + 1)
        ]
        self.y = [
            [0.0 for _ in range(MAX_COMP_NO + 1)] for _ in range(MAX_MOTION_NO + 1)
        ]
        self.z = [
            [0.0 for _ in range(MAX_COMP_NO + 1)] for _ in range(MAX_MOTION_NO + 1)
        ]
        self.o2 = [
            [0.0 for _ in range(MAX_COMP_NO + 1)] for _ in range(MAX_MOTION_NO + 1)
        ]
        self.o3 = [0.0] * (MAX_MOTION_NO + 1)
        self.teach = [
            [0.0 for _ in range(MAX_MOTION_NO + 1)]
            for _ in range(MAX_SAMPLE_NO * MAX_MOTION_NO + 1)
        ]
        self.jw = [0.0] * (
                (1 + EMG_MAX_CH * (EMG_MAX_CH + 3) // 2) * MAX_COMP_NO * MAX_MOTION_NO + 1
        )
        self.w = [
            [[0.0 for _ in range(MAX_COMP_NO + 1)] for _ in range(MAX_MOTION_NO + 1)]
            for _ in range(1 + EMG_MAX_CH * (EMG_MAX_CH + 3) // 2 + 1)
        ]
        self.dw = [
            [[0.0 for _ in range(MAX_COMP_NO + 1)] for _ in range(MAX_MOTION_NO + 1)]
            for _ in range(1 + EMG_MAX_CH * (EMG_MAX_CH + 3) // 2 + 1)
        ]
        self.wdt = [0.0] * (
                (1 + EMG_MAX_CH * (EMG_MAX_CH + 3) // 2) * MAX_COMP_NO * MAX_MOTION_NO + 1
        )
        self.wdt_pre = [0.0] * (
                (1 + EMG_MAX_CH * (EMG_MAX_CH + 3) // 2) * MAX_COMP_NO * MAX_MOTION_NO + 1
        )

        self.weightdata = "bitalino_sample/data/weight-3move-autocolor.csv"
        # self.learndata = 'bitalino_sample/data/learnvec.csv'
        # self.classsam = 'bitalino_sample/data/classvec.csv'
        # self.endata = 'bitalino_sample/data/Energy.csv'
        self.resdata = "bitalino_sample/data/res.csv"
        self.resonly_data = "bitalino_sample/data/res_only.csv"

        # 
        self.ser = serial.Serial('COM10',9600,timeout=None)  # データ取得だけの時一時的にコントアウト

        # 重みファイルのオープン
        with open(self.weightdata, "r", newline="") as fp_wt_pre:
            lines = fp_wt_pre.readline()
            lines_one = lines.split(",")
            # 重みの読み込み
            num_count = 0
            for k in range(self.motion_no + 1):
                for j in range(self.comp_no + 1):
                    for i in range(1 + self.emg_ch * (self.emg_ch + 3) // 2 + 1):
                        # ★w[i][k][j] = float(lines.pop(0))
                        row = lines_one[num_count]
                        # row = next(lines)
                        self.w[i][k][j] = float(row)
                        num_count += 1

    # パターン識別処理用関数
    def pttern_judg(self, data):
        emg_ch = self.emg_ch  # 使用EMGチャンネル数
        motion_no = self.motion_no  # 対象動作数
        comp_no = self.comp_no  # コンポーネント数
        sample_no = self.sample_no

        with open(self.resdata, "w", newline="") as fp_res, open(
                self.resonly_data, "w", newline=""
        ) as fp_resonly:
            # for ptn in range(1, self.class_dt_size + 1):

            dummy0 = [0.0] * (emg_ch + 1)
            self.forearm_motion = 0

            # ダミーデータをファイルから読み込む
            # line = fp_data.readline()
            # line_num = line.split(',')

            line_num = data[-1, :]  # 49はrad10_bufferが(50,4)のデータで，最後のデータだけほしいから

            line_sum = sum(data[-1, :])
            if line_sum == 0:
                self.forearm_motion = 0

            else:
                for i in range(1, emg_ch + 1):
                    dummy0[i] = float(line_num[i - 1])
                    # dummy0[i] = float(fp_data.readline().strip())

                # ダミーデータの合計を計算して正規化処理
                dummy_sum = sum(dummy0[1:])
                for i in range(1, emg_ch + 1):
                    self.dummy[i] = dummy0[i] / dummy_sum

                # LLGMN用入力ベクトルの変換
                x = [0.0] * (1 + emg_ch * (emg_ch + 3) // 2 + 1 + 1)
                x[1] = 1.0
                for i in range(2, 2 + emg_ch):
                    x[i] = self.dummy[i - 1]
                i3 = 2 + emg_ch
                for i1 in range(emg_ch):
                    for i2 in range(i1, emg_ch):
                        x[i3] = x[i1 + 2] * x[i2 + 2]
                        i3 += 1

                y = [[0.0] * (comp_no + 1) for _ in range(motion_no + 1)]
                z = [[0.0] * (comp_no + 1) for _ in range(motion_no + 1)]
                o2 = [[0.0] * (comp_no + 1) for _ in range(motion_no + 1)]
                o3 = [0.0] * (motion_no + 1)

                for k in range(1, motion_no):
                    for i in range(1, comp_no + 1):
                        y[k][i] = 0.0
                        for j in range(1, 2 + emg_ch * (emg_ch + 3) // 2):
                            y[k][i] += self.w[j][k][i] * x[j]

                k = motion_no
                for i in range(1, comp_no):
                    y[k][i] = 0.0
                    for j in range(1, 2 + emg_ch * (emg_ch + 3) // 2):
                        y[k][i] += self.w[j][k][i] * x[j]

                y[motion_no][comp_no] = 0.0
                g_s_sum = 0.0

                for i in range(1, motion_no + 1):
                    for j in range(1, comp_no + 1):
                        z[i][j] = math.exp(y[i][j])
                        g_s_sum += z[i][j]

                for i in range(1, motion_no + 1):
                    for j in range(1, comp_no + 1):
                        o2[i][j] = z[i][j] / g_s_sum

                for i in range(1, motion_no + 1):
                    o3[i] = 0.0
                    for j in range(1, comp_no + 1):
                        o3[i] += o2[i][j]

                entropy = 0.0
                for i in range(1, motion_no + 1):
                    entropy -= o3[i] * math.log(o3[i]) / math.log(2.0)

                fp_res.write(str(entropy))
                o3_max = 0

                for i in range(1, motion_no + 1):
                    if o3_max < o3[i]:
                        o3_max = o3[i]
                        self.forearm_motion = i

                fp_res.write(str(entropy) + " " + str(self.forearm_motion) + "\n")
                fp_resonly.write(str(self.forearm_motion) + "\n")
            # print(self.forearm_motion)

            if self.forearm_motion == 0:  # データ取得時一時的にコントアウト 無力
                move = "z"
            elif self.forearm_motion == 1:  # close
                move = "c"
            elif self.forearm_motion == 2:  # open
                move = "o"

            if self.forearm_motion == 3:  # 回外
                move = 0
                self.change_color += 1
                if self.change_color == 4:
                    self.count_color += 1
                    if self.count_color == 3:  # ハンドで追従する色の変更，赤→黄→緑→青
                        self.count_color = 0

            if self.count_color == 0:  # red
                color_choice = "R"
            elif self.count_color == 1:  # yellow
                color_choice = "Y"
            elif self.count_color == 2:  # green
                color_choice = "G"
            elif self.count_color == 3:  # blue
                color_choice = "B"

            # if move != 0:
            #     self.ser.write(bytes(color_choice, 'utf-8'))
            #     self.ser.write(bytes(move, 'utf-8'))
            #     print(color_choice,move)
            #     self.change_color = 0

            # time.sleep(0.07)

    # # パターン識別部の初期化用関数
    # def llgmn_ini(self):
    #     # 重みファイルのオープン
    #     with open(self.weightdata, 'r', newline='') as fp_wt_pre:
    #         lines = fp_wt_pre.readline()
    #         lines_one = lines.split(',')
    #         # 重みの読み込み
    #         num_count = 0
    #         for k in range(self.motion_no + 1):
    #             for j in range(self.comp_no + 1):
    #                 for i in range(1 + self.emg_ch * (self.emg_ch + 3) // 2 + 1):
    #                     #★w[i][k][j] = float(lines.pop(0))
    #                     row = lines_one[num_count]
    #                     # row = next(lines)
    #                     self.w[i][k][j] = float(row)
    #                     num_count += 1


if __name__ == "__main__":
    main = SerialAutocolor()
