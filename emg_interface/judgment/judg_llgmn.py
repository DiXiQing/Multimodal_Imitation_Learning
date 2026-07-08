import math


class JudgLLGMN:
    def __init__(
            self,
            MAX_COMP_NO=3,  # LLGMNの最大コンポーネント数
            EMG_MAX_CH=4,  # 最大次元数
            MAX_MOTION_NO=6,  # 最大動作数
            MAX_SAMPLE_NO=100,  # 最大サンプル数
    ):
        super().__init__()

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

        self.weightdata = "bitalino_sample/data/weight-6move.csv"
        self.learndata = "bitalino_sample/data/learnvec.csv"
        self.classsam = "bitalino_sample/data/classvec.csv"
        self.endata = "bitalino_sample/data/Energy.csv"
        self.resdata = "bitalino_sample/data/res.csv"
        self.resonly_data = "bitalino_sample/data/res_only.csv"

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

        # ここでlearn_llgmn()とllgmn_ini()を定義して実行
        # learn_llgmn()
        # llgmn_ini()

        dummy0 = [0.0] * (emg_ch + 1)
        self.forearm_motion = 0

        # ダミーデータをファイルから読み込む
        # line = fp_data.readline()
        # line_num = line.split(',')

        line_num = data[49]  # 49はrad10_bufferが(50,4)のデータで，最後のデータだけほしいから

        line_sum = sum(data[49])
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

            o3_max = 0

            for i in range(1, motion_no + 1):
                if o3_max < o3[i]:
                    o3_max = o3[i]
                    self.forearm_motion = i


if __name__ == "__main__":
    main = JudgLLGMN()
