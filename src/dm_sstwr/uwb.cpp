#include "dm_sstwr/uwb.h"
#include "dw3000.h"

dwt_config_t config = {
    5,                /* Channel number. */
    DWT_PLEN_128,     /* Preamble length. Used in TX only. */
    DWT_PAC8,         /* Preamble acquisition chunk size. Used in RX only. */
    9,                /* TX preamble code. Used in TX only. */
    9,                /* RX preamble code. Used in RX only. */
    1,                /* Non-standard 8 symbol SFD. */
    DWT_BR_6M8,       /* Data rate. */
    DWT_PHRMODE_STD,  /* PHY header mode. */
    DWT_PHRRATE_STD,  /* PHY header rate. */
    (129 + 8 - 8),    /* SFD timeout. */
    DWT_STS_MODE_OFF, /* STS disabled. */
    DWT_STS_LEN_64,
    DWT_PDOA_M0};
extern dwt_txconfig_t txconfig_options;

uint8_t tx_msg[MSG_LEN] = {0x41, 0x88, 0, 0xCA, 0xDE,
                           0, 0, UID, 0, FUNC_CODE_RANGE_CMD};
uint8_t rx_msg[MSG_LEN];

uint8_t frame_seq_nb = 0;
uint8_t cycle_seq_nb = 0;
uint32_t status_reg = 0;
int ret = 0;

float latest_distances[PAIR_COUNT];
float latest_rx_power_dbm[PAIR_COUNT];
float latest_fp_power_dbm[PAIR_COUNT];
bool latest_distance_valid[PAIR_COUNT];
bool latest_distance_recovered[PAIR_COUNT];
uint8_t latest_distance_seq[PAIR_COUNT];
unsigned long latest_distance_ms[PAIR_COUNT];

static unsigned long last_collector_print_millis = 0;
static uint8_t polling_pair_index = 0;
static uint8_t next_request_id = 1;
static uint8_t current_cycle_mask_seq = 0;
static uint16_t current_cycle_measured_mask = 0;
static uint16_t current_cycle_no_response_mask = 0;
static uint16_t current_cycle_recovered_mask = 0;
static uint8_t last_completed_cycle_seq = 0;
static uint16_t last_completed_measured_mask = 0;
static uint16_t last_completed_no_response_mask = 0;
static uint16_t last_completed_recovered_mask = 0;

static const uint8_t node_uid[NUM_NODES] = {
    NODE_A_UID, NODE_B_UID, NODE_C_UID,
    NODE_D_UID, NODE_E_UID, NODE_F_UID};
static const char *const node_label[NUM_NODES] = {
    "A", "B", "C", "D", "E", "F"};
static bool gateway_candidate_reachable[NUM_NODES];
static float gateway_candidate_distance_m[NUM_NODES];

static const uint8_t pair_from[PAIR_COUNT] = {
    U1, U2, U2, U2, U2, U3, U4, U5, U3, U3, U4, U1, U1, U1, U1};

static const uint8_t pair_to[PAIR_COUNT] = {
    U2, U3, U4, U5, U6, U4, U5, U6, U5, U6, U6, U3, U4, U5, U6};

static const char *const pair_label[PAIR_COUNT] = {
    "AB", "BC", "BD", "BE", "BF", "CD", "DE", "EF",
    "CE", "CF", "DF", "AC", "AD", "AE", "AF"};

struct WallRecoveryRule
{
    uint8_t missing_pair_index;
    uint8_t first_height_pair_index;
    uint8_t baseline_pair_index;
    uint8_t second_height_pair_index;
};

// AB, CD, and EF are parallel vertical pairs. B, D, and F share the
// door-side baseline. Each rule recovers the link between the upper nodes.
static const WallRecoveryRule wall_recovery_rules[] = {
    {PAIR_AC_INDEX, PAIR_AB_INDEX, PAIR_BD_INDEX, PAIR_CD_INDEX},
    {PAIR_AE_INDEX, PAIR_AB_INDEX, PAIR_BF_INDEX, PAIR_EF_INDEX},
    {PAIR_CE_INDEX, PAIR_CD_INDEX, PAIR_DF_INDEX, PAIR_EF_INDEX},
};

static const uint8_t WALL_RECOVERY_RULE_COUNT =
    (uint8_t)(sizeof(wall_recovery_rules) /
              sizeof(wall_recovery_rules[0]));

static bool time_before(unsigned long now, unsigned long deadline)
{
    return (long)(now - deadline) < 0;
}

static bool is_known_uid(uint8_t uid)
{
    return uid == U1 || uid == U2 || uid == U3 ||
           uid == U4 || uid == U5 || uid == U6;
}

static int node_index_from_uid(uint8_t uid)
{
    for (int i = 0; i < NUM_NODES; ++i)
    {
        if (node_uid[i] == uid)
        {
            return i;
        }
    }
    return -1;
}

static const char *node_label_from_uid(uint8_t uid)
{
    const int node_index = node_index_from_uid(uid);
    return node_index >= 0 ? node_label[node_index] : "?";
}

static int find_pair_index(uint8_t uid_a, uint8_t uid_b)
{
    for (int i = 0; i < PAIR_COUNT; ++i)
    {
        if ((pair_from[i] == uid_a && pair_to[i] == uid_b) ||
            (pair_from[i] == uid_b && pair_to[i] == uid_a))
        {
            return i;
        }
    }
    return -1;
}

static bool pair_contains_uid(uint8_t pair_index, uint8_t uid)
{
    if (pair_index >= PAIR_COUNT)
    {
        return false;
    }
    return pair_from[pair_index] == uid || pair_to[pair_index] == uid;
}

static uint8_t other_pair_endpoint(uint8_t pair_index, uint8_t endpoint_uid)
{
    if (pair_index >= PAIR_COUNT)
    {
        return U0;
    }
    if (pair_from[pair_index] == endpoint_uid)
    {
        return pair_to[pair_index];
    }
    if (pair_to[pair_index] == endpoint_uid)
    {
        return pair_from[pair_index];
    }
    return U0;
}

static bool is_gateway_routed_pair(uint8_t pair_index)
{
    return pair_index < PAIR_COUNT &&
           !pair_contains_uid(pair_index, NODE_B_UID);
}

static void reset_gateway_candidates()
{
    for (int i = 0; i < NUM_NODES; ++i)
    {
        gateway_candidate_reachable[i] = false;
        gateway_candidate_distance_m[i] = NAN;
    }
}

static void mark_gateway_candidate(uint8_t pair_index, float meters)
{
    if (!pair_contains_uid(pair_index, NODE_B_UID))
    {
        return;
    }

    const uint8_t candidate_uid =
        other_pair_endpoint(pair_index, NODE_B_UID);
    const int candidate_index = node_index_from_uid(candidate_uid);
    if (candidate_index < 0 || candidate_uid != FIXED_GATEWAY_UID)
    {
        return;
    }

    gateway_candidate_reachable[candidate_index] = true;
    gateway_candidate_distance_m[candidate_index] =
        isfinite(meters) && meters > 0.0f ? meters : NAN;
}

struct GatewayCandidate
{
    uint8_t uid;
    float distance_to_b_m;
    bool pair_endpoint;
};

static uint8_t build_gateway_candidates(
    uint8_t pair_index,
    GatewayCandidate candidates[MAX_GATEWAY_CANDIDATES])
{
    const int candidate_index = node_index_from_uid(FIXED_GATEWAY_UID);
    if (candidate_index < 0 ||
        !gateway_candidate_reachable[candidate_index])
    {
        return 0;
    }

    candidates[0] = {
        FIXED_GATEWAY_UID,
        gateway_candidate_distance_m[candidate_index],
        pair_contains_uid(pair_index, FIXED_GATEWAY_UID),
    };
    return 1;
}

static uint8_t allocate_request_id()
{
    const uint8_t request_id = next_request_id;
    ++next_request_id;
    if (next_request_id == 0)
    {
        next_request_id = 1;
    }
    return request_id;
}

static uint8_t combine_failed_status(uint8_t first_status,
                                     uint8_t second_status)
{
    if (first_status == REPORT_STATUS_NO_RESPONSE ||
        second_status == REPORT_STATUS_NO_RESPONSE)
    {
        return REPORT_STATUS_NO_RESPONSE;
    }
    if (first_status == REPORT_STATUS_INVALID_RANGE ||
        second_status == REPORT_STATUS_INVALID_RANGE)
    {
        return REPORT_STATUS_INVALID_RANGE;
    }
    return REPORT_STATUS_LOCAL_ERROR;
}

static bool frame_is_for_me(const uint8_t *frame)
{
    return frame[MSG_DID_IDX] == UID || frame[MSG_DID_IDX] == BROADCAST_UID;
}

static bool frame_header_ok(const uint8_t *frame)
{
    return frame[0] == 0x41 && frame[1] == 0x88 &&
           frame[3] == 0xCA && frame[4] == 0xDE;
}

static bool is_completed_scan_distance(uint8_t pair_index)
{
    if (pair_index >= PAIR_COUNT ||
        !latest_distance_valid[pair_index] ||
        latest_distance_seq[pair_index] != last_completed_cycle_seq)
    {
        return false;
    }
    const uint16_t pair_mask = (uint16_t)(1U << pair_index);
    return ((last_completed_measured_mask |
             last_completed_recovered_mask) &
            pair_mask) != 0;
}

static uint16_t distance_to_cm(float meters)
{
    if (!(meters >= 0.0f) || meters > MAX_VALID_DISTANCE_M)
    {
        return REPORT_INVALID_CM;
    }

    float cm = meters * 100.0f;
    if (cm >= (float)REPORT_INVALID_CM)
    {
        return REPORT_INVALID_CM - 1;
    }
    return (uint16_t)(cm + 0.5f);
}

static uint16_t read_u16_le(const uint8_t *buf)
{
    return (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
}

static void write_u16_le(uint8_t *buf, uint16_t value)
{
    buf[0] = (uint8_t)(value & 0xFF);
    buf[1] = (uint8_t)(value >> 8);
}

static int16_t read_i16_le(const uint8_t *buf)
{
    return (int16_t)read_u16_le(buf);
}

static void write_i16_le(uint8_t *buf, int16_t value)
{
    write_u16_le(buf, (uint16_t)value);
}

static void read_signal_power_dbm(float *rx_power_dbm,
                                  float *fp_power_dbm)
{
    static const uint32_t DGC_DBG_REGISTER_ID = 0x30060UL;
    static const double IPATOV_PRF_64_CONSTANT_DB = 121.7;
    static const double FIRST_PATH_AMPLITUDE_SCALE = 4.0;

    *rx_power_dbm = NAN;
    *fp_power_dbm = NAN;

    // Only read the Ipatov fields used by the two power formulas.  The
    // generic dwt_readdiagnostics() call reads a 216-byte diagnostics block,
    // which unnecessarily keeps a successful ranging/report path waiting.
    const uint16_t accumulation_count =
        dwt_read16bitoffsetreg(IP_DIAG_12_ID, 0) & 0x0FFFU;
    const double n = (double)accumulation_count;
    if (n <= 0.0)
    {
        return;
    }

    double dgc_correction_db = 0.0;
    const bool dgc_enabled =
        (dwt_read8bitoffsetreg(DGC_CFG_ID, 0) &
         DGC_CFG_RX_TUNE_EN_BIT_MASK) != 0;
    if (dgc_enabled)
    {
        const uint8_t dgc_decision =
            (uint8_t)((dwt_read8bitoffsetreg(DGC_DBG_REGISTER_ID, 3) >> 4) &
                      0x07U);
        dgc_correction_db = 6.0 * (double)dgc_decision;
    }

    const double power =
        (double)(dwt_read32bitreg(IP_DIAG_1_ID) & 0x0001FFFFUL);
    if (power > 0.0)
    {
        // DW3000 RX_POWER, 64 MHz Ipatov preamble (User Manual 4.7.2).
        *rx_power_dbm = (float)(10.0 * log10(power * 2097152.0 / (n * n)) +
                                dgc_correction_db - IPATOV_PRF_64_CONSTANT_DB);
    }

    // F1/F2/F3 contain two fractional bits, so raw amplitudes are / 4.
    const double f1 =
        (double)(dwt_read32bitreg(IP_DIAG_2_ID) & 0x003FFFFFUL) /
        FIRST_PATH_AMPLITUDE_SCALE;
    const double f2 =
        (double)(dwt_read32bitreg(IP_DIAG_3_ID) & 0x003FFFFFUL) /
        FIRST_PATH_AMPLITUDE_SCALE;
    const double f3 =
        (double)(dwt_read32bitreg(IP_DIAG_4_ID) & 0x003FFFFFUL) /
        FIRST_PATH_AMPLITUDE_SCALE;
    const double fp_power = f1 * f1 + f2 * f2 + f3 * f3;
    if (fp_power > 0.0)
    {
        // DW3000 FP_POWER, 64 MHz Ipatov preamble (User Manual 4.7.1).
        *fp_power_dbm = (float)(10.0 * log10(fp_power / (n * n)) +
                                dgc_correction_db -
                                IPATOV_PRF_64_CONSTANT_DB);
    }
}

static void clear_radio_status()
{
    dwt_write32bitreg(SYS_STATUS_ID,
                      SYS_STATUS_TXFRS_BIT_MASK | SYS_STATUS_RXFCG_BIT_MASK |
                          SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
}

static void clear_buffer(uint8_t *buf, int len)
{
    for (int i = 0; i < len; ++i)
    {
        buf[i] = 0;
    }
}

static void prepare_frame(uint8_t func, uint8_t dest_uid)
{
    clear_buffer(tx_msg, MSG_LEN);
    tx_msg[0] = 0x41;
    tx_msg[1] = 0x88;
    tx_msg[MSG_SN_IDX] = frame_seq_nb;
    tx_msg[3] = 0xCA;
    tx_msg[4] = 0xDE;
    tx_msg[MSG_DID_IDX] = dest_uid;
    tx_msg[MSG_DID_IDX + 1] = 0;
    tx_msg[MSG_SID_IDX] = UID;
    tx_msg[MSG_SID_IDX + 1] = 0;
    tx_msg[MSG_FUNC_IDX] = func;
}

static uint16_t read_rx_frame(uint8_t *buf)
{
    clear_buffer(buf, MSG_LEN);
    uint16_t frame_len = (uint16_t)(dwt_read32bitreg(RX_FINFO_ID) & RXFLEN_MASK);
    if (frame_len > MSG_LEN)
    {
        frame_len = MSG_LEN;
    }
    dwt_readrxdata(buf, frame_len, 0);
    return frame_len;
}

static bool wait_for_tx_done()
{
    const unsigned long deadline_ms = millis() + TX_DONE_TIMEOUT_MS;

    while (time_before(millis(), deadline_ms))
    {
        status_reg = dwt_read32bitreg(SYS_STATUS_ID);

        if (status_reg & SYS_STATUS_TXFRS_BIT_MASK)
        {
            dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS_BIT_MASK);
            return true;
        }

        if (status_reg & (SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR))
        {
            dwt_write32bitreg(SYS_STATUS_ID,
                              SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
            return false;
        }

        yield();
    }

    dwt_forcetrxoff();
    clear_radio_status();
    return false;
}

static bool transmit_current_frame(bool ranging_frame)
{
    clear_radio_status();
    tx_msg[MSG_SN_IDX] = frame_seq_nb;
    dwt_writetxdata((uint16_t)MSG_LEN, tx_msg, 0);
    dwt_writetxfctrl((uint16_t)MSG_LEN, 0, ranging_frame ? 1 : 0);

    ret = dwt_starttx(DWT_START_TX_IMMEDIATE);
    if (ret != DWT_SUCCESS)
    {
        dwt_forcetrxoff();
        clear_radio_status();
        return false;
    }

    const bool ok = wait_for_tx_done();
    ++frame_seq_nb;
    return ok;
}

static bool receive_frame(unsigned long timeout_ms, uint64_t *rx_timestamp)
{
    const unsigned long deadline_ms = millis() + timeout_ms;
    clear_radio_status();
    dwt_rxenable(DWT_START_RX_IMMEDIATE);

    while (time_before(millis(), deadline_ms))
    {
        status_reg = dwt_read32bitreg(SYS_STATUS_ID);

        if (status_reg & SYS_STATUS_RXFCG_BIT_MASK)
        {
            dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);
            read_rx_frame(rx_msg);
            *rx_timestamp = get_rx_timestamp_u64();
            dwt_forcetrxoff();
            return true;
        }

        if (status_reg & (SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR))
        {
            dwt_write32bitreg(SYS_STATUS_ID,
                              SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
            dwt_rxenable(DWT_START_RX_IMMEDIATE);
        }

        yield();
    }

    dwt_forcetrxoff();
    clear_radio_status();
    return false;
}

static bool store_distance(uint8_t pair_index, float meters,
                           float rx_power_dbm, float fp_power_dbm,
                           uint8_t seq, bool recovered)
{
    if (pair_index >= PAIR_COUNT)
    {
        return false;
    }
    if (!isfinite(meters) || meters <= 0.0f ||
        meters > MAX_VALID_DISTANCE_M)
    {
        return false;
    }

    latest_distances[pair_index] = meters;
    latest_rx_power_dbm[pair_index] = rx_power_dbm;
    latest_fp_power_dbm[pair_index] = fp_power_dbm;
    latest_distance_valid[pair_index] = true;
    latest_distance_recovered[pair_index] = recovered;
    latest_distance_seq[pair_index] = seq;
    latest_distance_ms[pair_index] = millis();
    return true;
}

static void invalidate_distance(uint8_t pair_index)
{
    if (pair_index >= PAIR_COUNT)
    {
        return;
    }

    latest_distance_valid[pair_index] = false;
    latest_distance_recovered[pair_index] = false;
    latest_distances[pair_index] = NAN;
    latest_rx_power_dbm[pair_index] = NAN;
    latest_fp_power_dbm[pair_index] = NAN;
}

static uint16_t pair_bit(uint8_t pair_index)
{
    return (uint16_t)(1U << pair_index);
}

static bool is_wall_recovery_target(uint8_t pair_index)
{
    for (uint8_t i = 0; i < WALL_RECOVERY_RULE_COUNT; ++i)
    {
        if (wall_recovery_rules[i].missing_pair_index == pair_index)
        {
            return true;
        }
    }
    return false;
}

static bool try_wall_recovery_rule(const WallRecoveryRule &rule)
{
#if ENABLE_DETERMINISTIC_WALL_RECOVERY
    const uint16_t missing_bit = pair_bit(rule.missing_pair_index);
    if (!(current_cycle_no_response_mask & missing_bit) ||
        (current_cycle_recovered_mask & missing_bit))
    {
        return false;
    }

    const uint8_t source_pairs[] = {
        rule.first_height_pair_index,
        rule.baseline_pair_index,
        rule.second_height_pair_index,
    };
    uint16_t required_mask = 0;
    for (uint8_t pair_index : source_pairs)
    {
        required_mask |= pair_bit(pair_index);
    }
    if ((current_cycle_measured_mask & required_mask) != required_mask)
    {
        return false;
    }

    for (uint8_t pair_index : source_pairs)
    {
        // Geometry may only consume UWB measurements from this scan.
        if (!latest_distance_valid[pair_index] ||
            latest_distance_recovered[pair_index] ||
            latest_distance_seq[pair_index] != current_cycle_mask_seq)
        {
            return false;
        }
    }

    const float first_height =
        latest_distances[rule.first_height_pair_index];
    const float baseline = latest_distances[rule.baseline_pair_index];
    const float second_height =
        latest_distances[rule.second_height_pair_index];
    if (!isfinite(first_height) || !isfinite(baseline) ||
        !isfinite(second_height) || first_height <= 0.0f ||
        baseline <= 0.0f || second_height <= 0.0f)
    {
        return false;
    }

    const float recovered_distance =
        hypotf(baseline, first_height - second_height);
    if (!isfinite(recovered_distance) || recovered_distance <= 0.0f ||
        recovered_distance > MAX_VALID_DISTANCE_M)
    {
        return false;
    }

    if (!store_distance(rule.missing_pair_index, recovered_distance,
                        NAN, NAN, cycle_seq_nb, true))
    {
        return false;
    }
    current_cycle_recovered_mask |= missing_bit;
    return true;
#else
    (void)rule;
    return false;
#endif
}

static void recover_wall_links_from_current_cycle()
{
    for (uint8_t i = 0; i < WALL_RECOVERY_RULE_COUNT; ++i)
    {
        try_wall_recovery_rule(wall_recovery_rules[i]);
    }
}

static bool send_range_response(const uint8_t *poll_frame,
                                uint64_t poll_rx_ts)
{
    const uint8_t initiator_uid = poll_frame[MSG_SID_IDX];
    const uint8_t pair_index = poll_frame[MSG_PAIR_IDX];

    if (!is_known_uid(initiator_uid) || pair_index >= PAIR_COUNT)
    {
        return false;
    }
    if (!pair_contains_uid(pair_index, UID) ||
        !pair_contains_uid(pair_index, initiator_uid) ||
        poll_frame[MSG_FROM_UID_IDX] != initiator_uid ||
        poll_frame[MSG_TO_UID_IDX] != UID)
    {
        return false;
    }

    prepare_frame(FUNC_CODE_RANGE_RESP, initiator_uid);
    tx_msg[MSG_PAIR_IDX] = pair_index;
    tx_msg[MSG_FROM_UID_IDX] = initiator_uid;
    tx_msg[MSG_TO_UID_IDX] = UID;

    const uint32_t resp_tx_time =
        (uint32_t)((poll_rx_ts +
                    ((uint64_t)POLL_RX_TO_RESP_TX_DLY_UUS * UUS_TO_DWT_TIME)) >>
                   8);
    dwt_setdelayedtrxtime(resp_tx_time);

    const uint64_t resp_tx_ts =
        (((uint64_t)(resp_tx_time & 0xFFFFFFFEUL)) << 8) + TX_ANT_DLY;
    resp_msg_set_ts(&tx_msg[RESP_MSG_POLL_RX_TS_IDX], poll_rx_ts);
    resp_msg_set_ts(&tx_msg[RESP_MSG_RESP_TX_TS_IDX], resp_tx_ts);

    clear_radio_status();
    tx_msg[MSG_SN_IDX] = frame_seq_nb;
    dwt_writetxdata((uint16_t)MSG_LEN, tx_msg, 0);
    dwt_writetxfctrl((uint16_t)MSG_LEN, 0, 1);

    ret = dwt_starttx(DWT_START_TX_DELAYED);
    if (ret != DWT_SUCCESS)
    {
        dwt_forcetrxoff();
        clear_radio_status();
        return false;
    }

    const bool ok = wait_for_tx_done();
    ++frame_seq_nb;
    return ok;
}

static bool range_to_peer(uint8_t pair_index, uint8_t target_uid,
                          float *meters_out, float *rx_power_dbm_out,
                          float *fp_power_dbm_out,
                          uint8_t *report_status_out)
{
    *report_status_out = REPORT_STATUS_LOCAL_ERROR;
    *rx_power_dbm_out = NAN;
    *fp_power_dbm_out = NAN;
    if (pair_index >= PAIR_COUNT || !is_known_uid(target_uid) ||
        !pair_contains_uid(pair_index, UID) ||
        !pair_contains_uid(pair_index, target_uid))
    {
        return false;
    }

    prepare_frame(FUNC_CODE_RANGE_POLL, target_uid);
    tx_msg[MSG_PAIR_IDX] = pair_index;
    tx_msg[MSG_FROM_UID_IDX] = UID;
    tx_msg[MSG_TO_UID_IDX] = target_uid;

    clear_radio_status();
    dwt_setrxaftertxdelay(POLL_TX_TO_RESP_RX_DLY_UUS);
    dwt_setrxtimeout(RESP_RX_TIMEOUT_UUS);
    tx_msg[MSG_SN_IDX] = frame_seq_nb;
    dwt_writetxdata((uint16_t)MSG_LEN, tx_msg, 0);
    dwt_writetxfctrl((uint16_t)MSG_LEN, 0, 1);

    ret = dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);
    if (ret != DWT_SUCCESS)
    {
        dwt_forcetrxoff();
        clear_radio_status();
        return false;
    }
    ++frame_seq_nb;

    const unsigned long deadline_ms = millis() + TX_DONE_TIMEOUT_MS;
    while (time_before(millis(), deadline_ms))
    {
        status_reg = dwt_read32bitreg(SYS_STATUS_ID);

        if (status_reg & SYS_STATUS_RXFCG_BIT_MASK)
        {
            dwt_write32bitreg(SYS_STATUS_ID,
                              SYS_STATUS_RXFCG_BIT_MASK |
                                  SYS_STATUS_TXFRS_BIT_MASK);
            read_rx_frame(rx_msg);

            if (frame_header_ok(rx_msg) && frame_is_for_me(rx_msg) &&
                rx_msg[MSG_FUNC_IDX] == FUNC_CODE_RANGE_RESP &&
                rx_msg[MSG_PAIR_IDX] == pair_index &&
                rx_msg[MSG_SID_IDX] == target_uid &&
                rx_msg[MSG_FROM_UID_IDX] == UID &&
                rx_msg[MSG_TO_UID_IDX] == target_uid)
            {
                // Snapshot every value used by ranging before any optional
                // signal-quality diagnostics are read.
                const uint32_t poll_tx_ts = dwt_readtxtimestamplo32();
                const uint32_t resp_rx_ts = dwt_readrxtimestamplo32();
                uint32_t poll_rx_ts = 0;
                uint32_t resp_tx_ts = 0;
                resp_msg_get_ts(&rx_msg[RESP_MSG_POLL_RX_TS_IDX],
                                &poll_rx_ts);
                resp_msg_get_ts(&rx_msg[RESP_MSG_RESP_TX_TS_IDX],
                                &resp_tx_ts);

                const int32_t rtd_init = (int32_t)(resp_rx_ts - poll_tx_ts);
                const int32_t rtd_resp = (int32_t)(resp_tx_ts - poll_rx_ts);
                const float clock_offset_ratio =
                    ((float)dwt_readclockoffset()) / (uint32_t)(1 << 26);

                const double tof =
                    ((rtd_init - rtd_resp * (1 - clock_offset_ratio)) / 2.0) *
                    DWT_TIME_UNITS;
                const double measured_distance = tof * SPEED_OF_LIGHT;

                if (!(measured_distance > 0.0) ||
                    measured_distance > MAX_VALID_DISTANCE_M)
                {
                    dwt_forcetrxoff();
                    clear_radio_status();
                    *report_status_out = REPORT_STATUS_INVALID_RANGE;
                    return false;
                }

                // A valid distance never depends on RSSI/FP availability.
                *meters_out = (float)measured_distance;
                *report_status_out = REPORT_STATUS_OK;

                float rx_power_dbm = NAN;
                float fp_power_dbm = NAN;
                read_signal_power_dbm(&rx_power_dbm, &fp_power_dbm);
                *rx_power_dbm_out = rx_power_dbm;
                *fp_power_dbm_out = fp_power_dbm;

                dwt_forcetrxoff();
                clear_radio_status();
                return true;
            }

            dwt_rxenable(DWT_START_RX_IMMEDIATE);
        }
        else if (status_reg & (SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR))
        {
            dwt_write32bitreg(SYS_STATUS_ID,
                              SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
            dwt_forcetrxoff();
            clear_radio_status();
            *report_status_out = REPORT_STATUS_NO_RESPONSE;
            return false;
        }

        yield();
    }

    dwt_forcetrxoff();
    clear_radio_status();
    *report_status_out = REPORT_STATUS_NO_RESPONSE;
    return false;
}

static bool send_report(uint8_t report_dest_uid, uint8_t pair_index,
                        float meters, float rx_power_dbm,
                        float fp_power_dbm, uint8_t report_status,
                        uint8_t request_id, uint8_t measurer_uid)
{
    if (!is_known_uid(report_dest_uid) || pair_index >= PAIR_COUNT ||
        (report_status == REPORT_STATUS_OK &&
         !pair_contains_uid(pair_index, measurer_uid)))
    {
        return false;
    }

    const bool ok = report_status == REPORT_STATUS_OK;
    prepare_frame(FUNC_CODE_RANGE_REPORT, report_dest_uid);
    tx_msg[MSG_PAIR_IDX] = pair_index;
    tx_msg[MSG_FROM_UID_IDX] = pair_from[pair_index];
    tx_msg[MSG_TO_UID_IDX] = pair_to[pair_index];
    tx_msg[MSG_STATUS_IDX] = report_status;
    write_u16_le(&tx_msg[MSG_DISTANCE_CM_IDX],
                 ok ? distance_to_cm(meters) : REPORT_INVALID_CM);
    const int16_t power_cdbm =
        (ok && isfinite(rx_power_dbm))
            ? (int16_t)roundf(rx_power_dbm * 100.0f)
            : INT16_MIN;
    write_i16_le(&tx_msg[MSG_RX_POWER_CDBM_IDX], power_cdbm);
    const int16_t fp_power_cdbm =
        (ok && isfinite(fp_power_dbm))
            ? (int16_t)roundf(fp_power_dbm * 100.0f)
            : INT16_MIN;
    write_i16_le(&tx_msg[MSG_FP_POWER_CDBM_IDX], fp_power_cdbm);
    tx_msg[MSG_REQUEST_ID_IDX] = request_id;
    tx_msg[MSG_MEASURER_UID_IDX] = ok ? measurer_uid : U0;
    return transmit_current_frame(false);
}

static bool wait_for_report(uint8_t pair_index,
                            uint8_t expected_reporter_uid,
                            uint8_t expected_measurer_uid,
                            uint8_t expected_request_id,
                            unsigned long report_wait_ms,
                            float *meters_out,
                            float *rx_power_dbm_out,
                            float *fp_power_dbm_out,
                            uint8_t *measurer_uid_out,
                            uint8_t *report_status_out)
{
    *meters_out = 0.0f;
    *rx_power_dbm_out = NAN;
    *fp_power_dbm_out = NAN;
    *measurer_uid_out = U0;
    *report_status_out = REPORT_STATUS_LOCAL_ERROR;
    if (pair_index >= PAIR_COUNT ||
        !is_known_uid(expected_reporter_uid) ||
        expected_request_id == 0)
    {
        return false;
    }

    const unsigned long deadline_ms = millis() + report_wait_ms;
    clear_radio_status();
    dwt_rxenable(DWT_START_RX_IMMEDIATE);

    while (time_before(millis(), deadline_ms))
    {
        status_reg = dwt_read32bitreg(SYS_STATUS_ID);

        if (status_reg & SYS_STATUS_RXFCG_BIT_MASK)
        {
            uint64_t rx_timestamp = 0;
            dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG_BIT_MASK);
            read_rx_frame(rx_msg);
            rx_timestamp = get_rx_timestamp_u64();

            if (frame_header_ok(rx_msg) && frame_is_for_me(rx_msg))
            {
                if (rx_msg[MSG_FUNC_IDX] == FUNC_CODE_RANGE_REPORT &&
                    rx_msg[MSG_DID_IDX] == UID &&
                    rx_msg[MSG_PAIR_IDX] == pair_index &&
                    rx_msg[MSG_SID_IDX] == expected_reporter_uid &&
                    rx_msg[MSG_FROM_UID_IDX] == pair_from[pair_index] &&
                    rx_msg[MSG_TO_UID_IDX] == pair_to[pair_index] &&
                    rx_msg[MSG_REQUEST_ID_IDX] ==
                        expected_request_id)
                {
                    const uint8_t received_status =
                        rx_msg[MSG_STATUS_IDX];
                    const uint16_t cm =
                        read_u16_le(&rx_msg[MSG_DISTANCE_CM_IDX]);
                    const uint8_t measurer_uid =
                        rx_msg[MSG_MEASURER_UID_IDX];
                    dwt_forcetrxoff();
                    clear_radio_status();

                    if (received_status > REPORT_STATUS_INVALID_RANGE)
                    {
                        return false;
                    }
                    *report_status_out = received_status;
                    if (received_status != REPORT_STATUS_OK)
                    {
                        return false;
                    }
                    if (cm == REPORT_INVALID_CM ||
                        !pair_contains_uid(pair_index, measurer_uid) ||
                        (expected_measurer_uid != U0 &&
                         measurer_uid != expected_measurer_uid))
                    {
                        *report_status_out = REPORT_STATUS_INVALID_RANGE;
                        return false;
                    }

                    const float meters = cm / 100.0f;
                    if (!(meters > 0.0f) ||
                        meters > MAX_VALID_DISTANCE_M)
                    {
                        *report_status_out = REPORT_STATUS_INVALID_RANGE;
                        return false;
                    }

                    *meters_out = meters;
                    *measurer_uid_out = measurer_uid;
                    const int16_t power_cdbm =
                        read_i16_le(&rx_msg[MSG_RX_POWER_CDBM_IDX]);
                    *rx_power_dbm_out =
                        power_cdbm == INT16_MIN ? NAN : power_cdbm / 100.0f;
                    const int16_t fp_power_cdbm =
                        read_i16_le(&rx_msg[MSG_FP_POWER_CDBM_IDX]);
                    *fp_power_dbm_out =
                        fp_power_cdbm == INT16_MIN
                            ? NAN
                            : fp_power_cdbm / 100.0f;
                    return true;
                }

                if (rx_msg[MSG_FUNC_IDX] == FUNC_CODE_RANGE_POLL &&
                    rx_msg[MSG_DID_IDX] == UID)
                {
                    send_range_response(rx_msg, rx_timestamp);
                }
            }

            dwt_rxenable(DWT_START_RX_IMMEDIATE);
        }
        else if (status_reg & (SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR))
        {
            dwt_write32bitreg(SYS_STATUS_ID,
                              SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
            dwt_rxenable(DWT_START_RX_IMMEDIATE);
        }

        yield();
    }

    dwt_forcetrxoff();
    clear_radio_status();
    *report_status_out = REPORT_STATUS_NO_RESPONSE;
    return false;
}

static bool request_remote_pair(uint8_t pair_index,
                                uint8_t initiator_uid,
                                float *meters_out,
                                float *rx_power_dbm_out,
                                float *fp_power_dbm_out,
                                uint8_t *measurer_uid_out,
                                uint8_t *report_status_out)
{
    *meters_out = 0.0f;
    *rx_power_dbm_out = NAN;
    *fp_power_dbm_out = NAN;
    *measurer_uid_out = U0;
    *report_status_out = REPORT_STATUS_LOCAL_ERROR;
    if (pair_index >= PAIR_COUNT)
    {
        return false;
    }

    const uint8_t responder_uid =
        other_pair_endpoint(pair_index, initiator_uid);
    if (!is_known_uid(initiator_uid) ||
        !is_known_uid(responder_uid) || initiator_uid == UID)
    {
        return false;
    }

    const uint8_t request_id = allocate_request_id();
    prepare_frame(FUNC_CODE_RANGE_CMD, initiator_uid);
    tx_msg[MSG_PAIR_IDX] = pair_index;
    tx_msg[MSG_FROM_UID_IDX] = initiator_uid;
    tx_msg[MSG_TO_UID_IDX] = responder_uid;
    tx_msg[MSG_STATUS_IDX] = REPORT_STATUS_OK;
    tx_msg[MSG_REQUEST_ID_IDX] = request_id;
    tx_msg[MSG_MEASURER_UID_IDX] = U0;

    if (!transmit_current_frame(false))
    {
        return false;
    }

    return wait_for_report(
        pair_index, initiator_uid, initiator_uid, request_id,
        POLLING_REPORT_WAIT_MS, meters_out, rx_power_dbm_out,
        fp_power_dbm_out, measurer_uid_out, report_status_out);
}

static bool attempt_gateway_pair(uint8_t pair_index,
                                 uint8_t initiator_uid,
                                 float *meters_out,
                                 float *rx_power_dbm_out,
                                 float *fp_power_dbm_out,
                                 uint8_t *measurer_uid_out,
                                 uint8_t *report_status_out)
{
    if (initiator_uid == UID)
    {
        const uint8_t responder_uid =
            other_pair_endpoint(pair_index, initiator_uid);
        if (!is_known_uid(responder_uid))
        {
            *report_status_out = REPORT_STATUS_LOCAL_ERROR;
            return false;
        }

        const bool ok = range_to_peer(
            pair_index, responder_uid, meters_out, rx_power_dbm_out,
            fp_power_dbm_out, report_status_out);
        *measurer_uid_out = ok ? UID : U0;
        return ok;
    }

    return request_remote_pair(
        pair_index, initiator_uid, meters_out, rx_power_dbm_out,
        fp_power_dbm_out, measurer_uid_out, report_status_out);
}

static bool collect_gateway_pair(uint8_t pair_index,
                                 float *meters_out,
                                 float *rx_power_dbm_out,
                                 float *fp_power_dbm_out,
                                 uint8_t *measurer_uid_out,
                                 uint8_t *report_status_out)
{
    *report_status_out = REPORT_STATUS_LOCAL_ERROR;
    if (UID == NODE_B_UID || !is_known_uid(UID) ||
        !is_gateway_routed_pair(pair_index))
    {
        return false;
    }

    // When the selected gateway is an endpoint, let it measure first. A
    // non-endpoint gateway asks each endpoint in the normal pair order.
    const uint8_t first_initiator =
        pair_contains_uid(pair_index, UID) ? UID : pair_from[pair_index];
    const uint8_t second_initiator =
        first_initiator == pair_from[pair_index]
            ? pair_to[pair_index]
            : pair_from[pair_index];

    uint8_t first_status = REPORT_STATUS_LOCAL_ERROR;
    bool ok = attempt_gateway_pair(
        pair_index, first_initiator, meters_out,
        rx_power_dbm_out, fp_power_dbm_out, measurer_uid_out,
        &first_status);
    if (ok)
    {
        *report_status_out = REPORT_STATUS_OK;
        return true;
    }

    delay(POLLING_DIRECTION_RETRY_GAP_MS);
    uint8_t second_status = REPORT_STATUS_LOCAL_ERROR;
    ok = attempt_gateway_pair(
        pair_index, second_initiator, meters_out,
        rx_power_dbm_out, fp_power_dbm_out, measurer_uid_out,
        &second_status);
    if (ok)
    {
        *report_status_out = REPORT_STATUS_OK;
        return true;
    }

    *measurer_uid_out = U0;
    *report_status_out =
        combine_failed_status(first_status, second_status);
    return false;
}

static bool request_pair_via_gateway(uint8_t pair_index,
                                     uint8_t gateway_uid,
                                     float *meters_out,
                                     float *rx_power_dbm_out,
                                     float *fp_power_dbm_out,
                                     uint8_t *measurer_uid_out,
                                     uint8_t *report_status_out)
{
    *report_status_out = REPORT_STATUS_LOCAL_ERROR;
    if (UID != NODE_B_UID || gateway_uid != FIXED_GATEWAY_UID ||
        !is_known_uid(gateway_uid) ||
        !is_gateway_routed_pair(pair_index))
    {
        return false;
    }

    const uint8_t request_id = allocate_request_id();
    prepare_frame(FUNC_CODE_GATEWAY_CMD, gateway_uid);
    tx_msg[MSG_PAIR_IDX] = pair_index;
    tx_msg[MSG_FROM_UID_IDX] = pair_from[pair_index];
    tx_msg[MSG_TO_UID_IDX] = pair_to[pair_index];
    tx_msg[MSG_STATUS_IDX] = REPORT_STATUS_OK;
    tx_msg[MSG_REQUEST_ID_IDX] = request_id;
    tx_msg[MSG_MEASURER_UID_IDX] = U0;

    if (!transmit_current_frame(false))
    {
        return false;
    }

    return wait_for_report(
        pair_index, gateway_uid, U0, request_id,
        GATEWAY_REPORT_WAIT_MS, meters_out, rx_power_dbm_out,
        fp_power_dbm_out, measurer_uid_out, report_status_out);
}

static const char *report_status_label(uint8_t status)
{
    switch (status)
    {
    case REPORT_STATUS_OK:
        return "OK";
    case REPORT_STATUS_NO_RESPONSE:
        return "NO_RESPONSE";
    case REPORT_STATUS_LOCAL_ERROR:
        return "LOCAL_ERROR";
    case REPORT_STATUS_INVALID_RANGE:
        return "INVALID_RANGE";
    default:
        return "UNKNOWN";
    }
}

static void print_gateway_choice(const char *event, uint8_t pair_index,
                                 uint8_t gateway_uid,
                                 bool include_status,
                                 uint8_t status)
{
    Serial.print(event);
    Serial.print(",pair=");
    Serial.print(pair_label[pair_index]);
    Serial.print(",gateway=");
    Serial.print(node_label_from_uid(gateway_uid));
    if (include_status)
    {
        Serial.print(",status=");
        Serial.print(report_status_label(status));
    }
    Serial.println();
}

static bool collect_pair_via_gateways(uint8_t pair_index,
                                      float *meters_out,
                                      float *rx_power_dbm_out,
                                      float *fp_power_dbm_out,
                                      uint8_t *measurer_uid_out,
                                      uint8_t *report_status_out)
{
    GatewayCandidate candidates[MAX_GATEWAY_CANDIDATES];
    const uint8_t candidate_count =
        build_gateway_candidates(pair_index, candidates);

    *report_status_out = REPORT_STATUS_NO_RESPONSE;
    *measurer_uid_out = U0;
    if (candidate_count == 0)
    {
        Serial.print("GATEWAY_NONE,pair=");
        Serial.print(pair_label[pair_index]);
        Serial.println(",status=NO_RESPONSE");
        return false;
    }

    for (uint8_t i = 0; i < candidate_count; ++i)
    {
        const uint8_t gateway_uid = candidates[i].uid;
        print_gateway_choice(
            i == 0 ? "GATEWAY_SELECT" : "GATEWAY_RETRY",
            pair_index, gateway_uid, false, REPORT_STATUS_OK);

        uint8_t attempt_status = REPORT_STATUS_LOCAL_ERROR;
        const bool ok = request_pair_via_gateway(
            pair_index, gateway_uid, meters_out, rx_power_dbm_out,
            fp_power_dbm_out, measurer_uid_out, &attempt_status);
        if (ok)
        {
            *report_status_out = REPORT_STATUS_OK;
            print_gateway_choice(
                "GATEWAY_SUCCESS", pair_index, gateway_uid,
                false, REPORT_STATUS_OK);
            return true;
        }

        print_gateway_choice(
            "GATEWAY_FAIL", pair_index, gateway_uid,
            true, attempt_status);
    }

    // A pair is unavailable after the fixed gateway attempt fails. Keep the
    // final status deterministic for mask generation.
    *report_status_out = REPORT_STATUS_NO_RESPONSE;
    *measurer_uid_out = U0;
    return false;
}

static void handle_gateway_command(const uint8_t *frame)
{
    // Snapshot command fields because nested ranging overwrites rx_msg.
    const uint8_t pair_index = frame[MSG_PAIR_IDX];
    const uint8_t request_id = frame[MSG_REQUEST_ID_IDX];

    if (UID != FIXED_GATEWAY_UID || !is_known_uid(UID) ||
        frame[MSG_DID_IDX] != UID ||
        frame[MSG_SID_IDX] != NODE_B_UID ||
        frame[MSG_FUNC_IDX] != FUNC_CODE_GATEWAY_CMD ||
        frame[MSG_STATUS_IDX] != REPORT_STATUS_OK || request_id == 0 ||
        !is_gateway_routed_pair(pair_index) ||
        frame[MSG_FROM_UID_IDX] != pair_from[pair_index] ||
        frame[MSG_TO_UID_IDX] != pair_to[pair_index])
    {
        return;
    }

    float meters = 0.0f;
    float rx_power_dbm = NAN;
    float fp_power_dbm = NAN;
    uint8_t measurer_uid = U0;
    uint8_t report_status = REPORT_STATUS_LOCAL_ERROR;

    delay(POLLING_WORKER_SETTLE_MS);
    collect_gateway_pair(
        pair_index, &meters, &rx_power_dbm, &fp_power_dbm,
        &measurer_uid, &report_status);
    send_report(
        NODE_B_UID, pair_index, meters, rx_power_dbm, fp_power_dbm,
        report_status, request_id, measurer_uid);
}

static void print_setup_roster()
{
    Serial.println(APP_NAME);
    Serial.print("ROLE,");
    Serial.print(ROLE_LABEL);
    Serial.print(",");
    Serial.print(ROLE_NAME);
    Serial.print(",UID,");
    Serial.print(UID);
    Serial.print(",TAG,0x");
    if (TAG_ID < 0x10)
    {
        Serial.print("0");
    }
    Serial.println(TAG_ID, HEX);
    Serial.println("MODE,U2_CENTERED_POLLING");
    Serial.println("PAIR_ORDER,AB,BC,BD,BE,BF,CD,DE,EF,CE,CF,DF,AC,AD,AE,AF");
#ifdef COLLECTOR_NODE
    Serial.print("COLLECTOR,B,serial_output=TABULAR_COMPLETED_SCAN,period_ms=");
    Serial.print(COLLECTOR_PRINT_INTERVAL_MS);
    Serial.print(",stale_ms=");
    Serial.println(MAX_DISTANCE_AGE_MS);
    Serial.println(
        "GATEWAY_MODE,FIXED,gateway=C,"
        "detection=DIRECT_B_SS_TWR,rssi_threshold=OFF");
#if ENABLE_DETERMINISTIC_WALL_RECOVERY
    Serial.println(
        "WALL_RECOVERY,mode=DETERMINISTIC,trigger=NO_RESPONSE,"
        "formula=hypot(BASELINE|HEIGHT1-HEIGHT2)");
    Serial.println(
        "WALL_RULES,AC=AB|BD|CD,AE=AB|BF|EF,CE=CD|DF|EF");
#endif
#endif
    Serial.println("Setup over........");
}

static void print_collector_table()
{
#ifdef COLLECTOR_NODE
    const unsigned long now = millis();
    if (now - last_collector_print_millis < COLLECTOR_PRINT_INTERVAL_MS)
    {
        return;
    }
    last_collector_print_millis = now;

    Serial.println("\n=======================================================");
    Serial.print("[UWB POLLING DISTANCE] SCAN: ");
    Serial.print(last_completed_cycle_seq);
    Serial.print(" | NEXT: ");
    Serial.print(pair_label[polling_pair_index]);
    Serial.print(" | TIME: ");
    Serial.print(now);
    Serial.println(" ms");
    Serial.println("-------------------------------------------------------");
    Serial.println("| PAIR | DISTANCE(m) | RSSI(dBm) | STATUS | AGE(ms) |");
    Serial.println("-------------------------------------------------------");

    for (int i = 0; i < PAIR_COUNT; ++i)
    {
        Serial.print("|  ");
        Serial.print(pair_label[i]);
        Serial.print("  | ");

        if (is_completed_scan_distance((uint8_t)i))
        {
            Serial.print(latest_distances[i], 3);
            Serial.print("       | ");
            if (isfinite(latest_rx_power_dbm[i]))
            {
                Serial.print(latest_rx_power_dbm[i], 2);
            }
            else
            {
                Serial.print("nan");
            }
            Serial.print(latest_distance_recovered[i]
                             ? " | RECOVERED | "
                             : " | VALID | ");
            Serial.println(now - latest_distance_ms[i]);
        }
        else
        {
            Serial.println("nan         | nan       | TIMEOUT | nan");
        }
    }
    Serial.println("=======================================================");

    const uint16_t fresh_mask =
        last_completed_measured_mask | last_completed_recovered_mask;
    const uint16_t recovered_mask = last_completed_recovered_mask;
    Serial.print("Original_DIST,seq=");
    Serial.print(last_completed_cycle_seq);
    Serial.print(",ms=");
    Serial.print(now);
    for (int i = 0; i < PAIR_COUNT; ++i)
    {
        Serial.print(",");
        Serial.print(pair_label[i]);
        Serial.print("=");
        if (is_completed_scan_distance((uint8_t)i))
        {
            Serial.print(latest_distances[i], 3);
        }
        else
        {
            Serial.print("nan");
        }
    }
    for (int i = 0; i < PAIR_COUNT; ++i)
    {
        Serial.print(",rssi_");
        Serial.print(pair_label[i]);
        Serial.print("=");
        if (is_completed_scan_distance((uint8_t)i) &&
            isfinite(latest_rx_power_dbm[i]))
        {
            Serial.print(latest_rx_power_dbm[i], 2);
        }
        else
        {
            Serial.print("nan");
        }
    }
    Serial.print(",fresh_mask=0x");
    Serial.print(fresh_mask, HEX);
    Serial.print(",recovered_mask=0x");
    Serial.print(recovered_mask, HEX);
    Serial.print(",last_scan_seq=");
    Serial.print(last_completed_cycle_seq);
    Serial.print(",last_measured_mask=0x");
    Serial.print(last_completed_measured_mask, HEX);
    Serial.print(",last_no_response_mask=0x");
    Serial.print(last_completed_no_response_mask, HEX);
    Serial.print(",last_recovered_mask=0x");
    Serial.println(last_completed_recovered_mask, HEX);
#endif
}

static void handle_worker_frame(const uint8_t *frame, uint64_t rx_timestamp)
{
    if (!frame_header_ok(frame) || !frame_is_for_me(frame))
    {
        return;
    }

    const uint8_t func = frame[MSG_FUNC_IDX];
    const uint8_t pair_index = frame[MSG_PAIR_IDX];

    if (func == FUNC_CODE_RANGE_POLL && frame[MSG_DID_IDX] == UID)
    {
        send_range_response(frame, rx_timestamp);
        return;
    }

    if (func == FUNC_CODE_GATEWAY_CMD && frame[MSG_DID_IDX] == UID)
    {
        handle_gateway_command(frame);
        return;
    }

    if (func != FUNC_CODE_RANGE_CMD || frame[MSG_DID_IDX] != UID)
    {
        return;
    }
    // Snapshot these fields before range_to_peer() overwrites rx_msg.
    const uint8_t coordinator_uid = frame[MSG_SID_IDX];
    const uint8_t request_id = frame[MSG_REQUEST_ID_IDX];
    if (!is_known_uid(coordinator_uid) || coordinator_uid == UID ||
        (coordinator_uid != NODE_B_UID &&
         !is_gateway_routed_pair(pair_index)) ||
        frame[MSG_STATUS_IDX] != REPORT_STATUS_OK ||
        request_id == 0 || pair_index >= PAIR_COUNT ||
        frame[MSG_FROM_UID_IDX] != UID)
    {
        return;
    }

    const uint8_t target_uid = other_pair_endpoint(pair_index, UID);
    if (target_uid == U0 || frame[MSG_TO_UID_IDX] != target_uid)
    {
        return;
    }

    float meters = 0.0f;
    float rx_power_dbm = NAN;
    float fp_power_dbm = NAN;
    uint8_t report_status = REPORT_STATUS_LOCAL_ERROR;

    delay(POLLING_WORKER_SETTLE_MS);
    range_to_peer(pair_index, target_uid, &meters, &rx_power_dbm,
                  &fp_power_dbm, &report_status);
    send_report(coordinator_uid, pair_index, meters, rx_power_dbm,
                fp_power_dbm, report_status, request_id, UID);
}

void start_uwb()
{
    while (!dwt_checkidlerc())
    {
        UART_puts("IDLE FAILED\r\n");
        while (1)
        {
        }
    }

    if (dwt_initialise(DWT_DW_INIT) == DWT_ERROR)
    {
        UART_puts("INIT FAILED\r\n");
        while (1)
        {
        }
    }

    dwt_setleds(DWT_LEDS_DISABLE);

    if (dwt_configure(&config))
    {
        UART_puts("CONFIG FAILED\r\n");
        while (1)
        {
        }
    }

    dwt_configuretxrf(&txconfig_options);
    dwt_setrxantennadelay(RX_ANT_DLY);
    dwt_settxantennadelay(TX_ANT_DLY);
    dwt_setrxaftertxdelay(POLL_TX_TO_RESP_RX_DLY_UUS);
    dwt_setrxtimeout(RESP_RX_TIMEOUT_UUS);
    dwt_setlnapamode(DWT_LNA_ENABLE | DWT_PA_ENABLE);
    dwt_configciadiag(DW_CIA_DIAG_LOG_ALL);

    print_setup_roster();
}

void polling_master()
{
    const uint8_t pair_index = polling_pair_index;
    float meters = 0.0f;
    float rx_power_dbm = NAN;
    float fp_power_dbm = NAN;
    bool ok = false;
    bool pair_no_response = false;
    uint8_t measurer_uid = U0;
    uint8_t range_status = REPORT_STATUS_LOCAL_ERROR;

    if (pair_index == 0)
    {
        current_cycle_mask_seq = cycle_seq_nb;
        current_cycle_measured_mask = 0;
        current_cycle_no_response_mask = 0;
        current_cycle_recovered_mask = 0;
        reset_gateway_candidates();
    }

    if (UID == NODE_B_UID && is_gateway_routed_pair(pair_index))
    {
        ok = collect_pair_via_gateways(
            pair_index, &meters, &rx_power_dbm, &fp_power_dbm,
            &measurer_uid, &range_status);
    }
    else if (pair_from[pair_index] == UID ||
             pair_to[pair_index] == UID)
    {
        const uint8_t peer_uid =
            (pair_from[pair_index] == UID) ? pair_to[pair_index]
                                           : pair_from[pair_index];
        ok = range_to_peer(pair_index, peer_uid, &meters, &rx_power_dbm,
                           &fp_power_dbm, &range_status);
        measurer_uid = ok ? UID : U0;
    }
    else
    {
        ok = request_remote_pair(
            pair_index, pair_from[pair_index], &meters,
            &rx_power_dbm, &fp_power_dbm, &measurer_uid,
            &range_status);
    }

    pair_no_response =
        !ok && range_status == REPORT_STATUS_NO_RESPONSE;

    const uint16_t current_pair_bit = pair_bit(pair_index);
    if (ok)
    {
        const bool stored =
            store_distance(pair_index, meters, rx_power_dbm, fp_power_dbm,
                           cycle_seq_nb, false);
        if (stored)
        {
            current_cycle_measured_mask |= current_pair_bit;
            current_cycle_no_response_mask &=
                (uint16_t)~current_pair_bit;
            current_cycle_recovered_mask &=
                (uint16_t)~current_pair_bit;
            mark_gateway_candidate(pair_index, meters);
        }
        else
        {
            invalidate_distance(pair_index);
        }
    }
    else
    {
        if (pair_contains_uid(pair_index, NODE_B_UID) &&
            range_status == REPORT_STATUS_INVALID_RANGE)
        {
            // A valid RANGE_RESP proves direct B communication even when the
            // calculated distance is unusable for distance-based ordering.
            mark_gateway_candidate(pair_index, NAN);
        }
        if (is_wall_recovery_target(pair_index) ||
            is_gateway_routed_pair(pair_index))
        {
            // A failed target or gateway request must not fall back to an
            // older cached distance.
            invalidate_distance(pair_index);
        }
        if (pair_no_response)
        {
            current_cycle_no_response_mask |= current_pair_bit;
        }
    }

    // Re-evaluate all rules after every pair. This also handles CE, whose
    // required DF measurement occurs later in the unchanged polling order.
    recover_wall_links_from_current_cycle();

    bool cycle_completed = false;
    ++polling_pair_index;
    if (polling_pair_index >= PAIR_COUNT)
    {
        last_completed_cycle_seq = current_cycle_mask_seq;
        last_completed_measured_mask = current_cycle_measured_mask;
        last_completed_no_response_mask =
            current_cycle_no_response_mask;
        last_completed_recovered_mask = current_cycle_recovered_mask;
        polling_pair_index = 0;
        ++cycle_seq_nb;
        cycle_completed = true;
    }

    if (cycle_completed)
    {
        print_collector_table();
    }
    delay(POLLING_PAIR_GAP_MS);
}

void polling_node()
{
    uint64_t rx_timestamp = 0;
    if (receive_frame(POLLING_WORKER_RX_WINDOW_MS, &rx_timestamp))
    {
        handle_worker_frame(rx_msg, rx_timestamp);
    }
}

void initiator()
{
    polling_master();
}

void responder()
{
    polling_node();
}
