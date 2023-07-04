import csv
import sys
import numpy as np
import pandas as pd
import math


def aggregation_time_traffic(pl, ts, tau):
    def append_into_list(traffic_agg_DF_list, n_packet_per_agg_DF_list, traffic_agg_UF_list, n_packet_per_agg_UF_list,
                         sumPL_DF, sumPL_UF, n_pkt_per_agg_DF, n_pkt_per_agg_UF):
        traffic_agg_DF_list.append(sumPL_DF)
        n_packet_per_agg_DF_list.append(n_pkt_per_agg_DF)
        traffic_agg_UF_list.append(sumPL_UF)
        n_packet_per_agg_UF_list.append(n_pkt_per_agg_UF)

    pl_n_elem = len(pl)
    traffic_agg_DF_list = []
    n_packet_per_agg_DF_list = []
    traffic_agg_UF_list = []
    n_packet_per_agg_UF_list = []

    if pl_n_elem != 0:
        ts = (ts - ts[0]) * 1000
        sumPL_DF, sumPL_UF, n_pkt_per_agg_DF, n_pkt_per_agg_UF = 0, 0, 0, 0
        i, offset = 0, 1
        while i < pl_n_elem:
            if ts[i] <= offset * tau:
                if pl[i] < 0 or math.copysign(1, pl[i]) < 0:
                    sumPL_UF += abs(pl[i])
                    n_pkt_per_agg_UF += 1
                else:
                    sumPL_DF += pl[i]
                    n_pkt_per_agg_DF += 1
                if i == pl_n_elem - 1:
                    append_into_list(traffic_agg_DF_list, n_packet_per_agg_DF_list, traffic_agg_UF_list,
                                     n_packet_per_agg_UF_list,
                                     sumPL_DF, sumPL_UF, n_pkt_per_agg_DF, n_pkt_per_agg_UF)
                i += 1
            else:
                offset += 1
                # Update of array of aggregate
                append_into_list(traffic_agg_DF_list, n_packet_per_agg_DF_list, traffic_agg_UF_list,
                                 n_packet_per_agg_UF_list,
                                 sumPL_DF, sumPL_UF, n_pkt_per_agg_DF, n_pkt_per_agg_UF)
                sumPL_DF, sumPL_UF, n_pkt_per_agg_DF, n_pkt_per_agg_UF = 0, 0, 0, 0

    else:
        append_into_list(traffic_agg_DF_list, n_packet_per_agg_DF_list, traffic_agg_UF_list, n_packet_per_agg_UF_list,
                         0, 0, 0, 0)

    return np.array(traffic_agg_UF_list), np.array(traffic_agg_DF_list), np.array(n_packet_per_agg_UF_list), \
           np.array(n_packet_per_agg_DF_list)


def compute_summary(df, lookahead, agg_time_list, app):
    n_biflow = df.shape[0]
    n_packet = sum([len(df.L4_payload_bytes.values[i]) for i in range(n_biflow)])
    n_biflow_per_lookahead = sum([1 for i in range(n_biflow) if len(df.L4_payload_bytes.values[i]) > lookahead])
    n_agg_list = []
    n_flows_agg_lookahead_list = []
    for i in agg_time_list:
        n_agg_list.append(sum([len(df['UF_volume_%sms' % i].values[j]) for j in
                               range(len(df['UF_volume_%sms' % i]))]))
        n_flows_agg_lookahead_list.append(sum([1 for j in range(len(df['UF_volume_%sms' % i]))
                                               if len(df['UF_volume_%sms' % i].values[j]) > lookahead]))

    agg_reduction_factor = [round(100 - (100 * float(n_agg_list[i]) / float(n_packet)), 2) for i in
                            range(len(agg_time_list))]
    agg_red_factor_lookahead = \
        [round(100 - (100 * float(n_flows_agg_lookahead_list[i]) / float(n_biflow_per_lookahead)), 2)
         for i in range(len(agg_time_list))]
    n_packet_list = [n_packet for _ in agg_time_list]
    n_biflow_list = [n_biflow for _ in agg_time_list]
    n_biflow_per_lookahead_list = [n_biflow_per_lookahead for _ in agg_time_list]
    filename = 'summary_%s_time_aggregate.csv' % app
    rows_per_csv = zip(n_biflow_list, n_biflow_per_lookahead_list, n_packet_list, agg_time_list, n_agg_list,
                       agg_reduction_factor, n_flows_agg_lookahead_list, agg_red_factor_lookahead)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['N_biflow', 'N_biflow_per_lookahead_%s' % lookahead, 'N_packet', 'Agg_time(ms)', 'N_agg',
                         'Agg_reduction_values%', 'N_bf_agg_per_lookahead_%s' % lookahead,
                         'Agg_reduction_factor%_per_lookahead_' + str(lookahead)])
        writer.writerows(rows_per_csv)
    """iat_UF = [iat for i in range(len(df.UF_iat_micros)) for iat in df.UF_iat_micros.values[i] if iat > 0]
    iat_mean_UF = np.mean(iat_UF) / 1000
    iat_std_UF = np.std(iat_UF) / 1000
    iat_DF = [iat for i in range(len(df.DF_iat_micros)) for iat in df.DF_iat_micros.values[i] if iat > 0]
    iat_mean_DF = np.mean(iat_DF) / 1000
    lookahead_in_time_mean_UF = iat_mean_UF * lookahead
    lookahead_in_time_mean_DF = iat_mean_DF * lookahead"""


def test_samets_compute_aggregation_time_traffic():
    pl = np.array([1400, -1400, -1400, 1400, -1400, 1400, -300, 1400, -1400, 300, 300])
    ts = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    agg_UF, agg_DF, n_pkt_UF, n_pkt_DF = aggregation_time_traffic(pl, ts, 2000)
    print('UF:', agg_UF, '\n', 'DF', agg_DF, '\n', 'n_pkt_UF:', n_pkt_UF, '\n', 'n_pkt_DF:', n_pkt_DF, '\n')


def test_diffts_compute_aggregation_time_traffic():
    pl = np.array([1400, -1400, -1400, 1400, -1400, 1400, -300, 1400, -1400, 300, 300])
    ts = np.array([0, 0, 1, 2, 2, 3, 4, 5, 7, 8, 9])
    agg_UF, agg_DF, n_pkt_UF, n_pkt_DF = aggregation_time_traffic(pl, ts, 9000)
    print('UF:', agg_UF, '\n', 'DF', agg_DF, '\n', 'n_pkt_UF:', n_pkt_UF, '\n', 'n_pkt_DF:', n_pkt_DF, '\n')


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage:', sys.argv[0], '<DATASET_FILENAME>', '<AGGREGATION TIME LIST>', '<APP>',
              '<PRINT_SUMMARY [TRUE, FALSE]>', '<LOOKAHEAD>')
        sys.exit(1)
    dataset_filename = sys.argv[1]
    agg_time_list = [int(time) for time in sys.argv[2].split(',')]
    app = sys.argv[3]
    data_in_df = pd.read_parquet(dataset_filename)
    data_app_df = data_in_df.loc[data_in_df['BF_label'] == app]
    if data_app_df.shape[0] == 0:
        apps = data_in_df.BF_label.unique()
        print(app, 'is not a valid app! Choose', apps)
        sys.exit(1)
    del data_in_df
    test = False
    if test:
        print('Some tests')
        test_samets_compute_aggregation_time_traffic()
        test_diffts_compute_aggregation_time_traffic()
    print('Start aggregation')
    for i in agg_time_list:
        print('Aggregation for %sms' % i)
        data_app_df['UF_volume_%sms' % i], data_app_df['DF_volume_%sms' % i], data_app_df['UF_N_packets_%sms' % i], \
        data_app_df['DF_N_packets_%sms' % i] = zip(*data_app_df.apply(
            lambda x: aggregation_time_traffic(x.L4_payload_bytes_dir, x.timestamp, i), axis=1))

    summary = sys.argv[4].lower() == 'true'
    lookahead = int(sys.argv[5]) if len(sys.argv) == 6 else 1
    if summary:
        columns_UF = ['UF_volume_%sms' % i for i in agg_time_list] + ['UF_N_packets_%sms' % i for i in agg_time_list]
        columns_DF = ['DF_volume_%sms' % i for i in agg_time_list] + ['DF_N_packets_%sms' % i for i in agg_time_list]
        selected_columns = columns_UF + columns_DF + ['L4_payload_bytes']
        selected_df = data_app_df[selected_columns]
        compute_summary(selected_df, lookahead, agg_time_list, app)

    filename_token = dataset_filename.split('.')
    filename_out = filename_token[0] + '_time_aggregate.' + filename_token[1]
    print('Saving new dataframe')
    data_app_df.to_parquet(filename_out)
    print('End!')
