#pragma once
#ifndef UWB_H
#define UWB_H

#include "dw3000.h"

#define NUM_NODES 6
#define UWB_RST 27
#define UWB_IRQ 34
#define UWB_SS 4

#define TAG_NONE 0x00
#define TAG_FIREFIGHTER_C 0x01
#define TAG_FIREFIGHTER_D 0x02
#define TAG_FIREFIGHTER_E 0x03
#define TAG_FIREFIGHTER_F 0x04

#define U0 0
#define U1 10
#define U2 14
#define U3 18
#define U4 22
#define U5 26
#define U6 30
#define BROADCAST_UID 0xFF

#define NODE_A_UID U1
#define NODE_B_UID U2
#define NODE_C_UID U3
#define NODE_D_UID U4
#define NODE_E_UID U5
#define NODE_F_UID U6

// B receives every pair that does not contain B through this node.
// Change only this UID when the report gateway changes.
#define REPORT_GATEWAY_UID NODE_E_UID

#ifdef MAIN_U1
#define APP_NAME "UWB DW3000 A EXTERNAL DRONE"
#define ROLE_LABEL "A"
#define ROLE_NAME "external_drone_worker"
#define TAG_ID TAG_NONE
#define UID NODE_A_UID
#elif defined(MAIN_U2)
#define APP_NAME "UWB DW3000 B POLLING MASTER"
#define ROLE_LABEL "B"
#define ROLE_NAME "internal_drone_polling_master"
#define TAG_ID TAG_NONE
#define UID NODE_B_UID
#define POLLING_MASTER
#define COLLECTOR_NODE
#elif defined(MAIN_U3)
#define APP_NAME "UWB DW3000 C FIREFIGHTER"
#define ROLE_LABEL "C"
#define ROLE_NAME "firefighter_worker"
#define TAG_ID TAG_FIREFIGHTER_C
#define UID NODE_C_UID
#elif defined(MAIN_U4)
#define APP_NAME "UWB DW3000 D FIREFIGHTER"
#define ROLE_LABEL "D"
#define ROLE_NAME "firefighter_worker"
#define TAG_ID TAG_FIREFIGHTER_D
#define UID NODE_D_UID
#elif defined(MAIN_U5)
#define APP_NAME "UWB DW3000 E FIREFIGHTER"
#define ROLE_LABEL "E"
#define ROLE_NAME "firefighter_worker"
#define TAG_ID TAG_FIREFIGHTER_E
#define UID NODE_E_UID
#elif defined(MAIN_U6)
#define APP_NAME "UWB DW3000 F FIREFIGHTER"
#define ROLE_LABEL "F"
#define ROLE_NAME "firefighter_worker"
#define TAG_ID TAG_FIREFIGHTER_F
#define UID NODE_F_UID
#else
#define APP_NAME "UWB DW3000 UNKNOWN"
#define ROLE_LABEL "?"
#define ROLE_NAME "unknown"
#define TAG_ID TAG_NONE
#define UID U0
#endif

#define TX_ANT_DLY 16385
#define RX_ANT_DLY 16385

#define FUNC_CODE_RANGE_CMD 0xE0
#define FUNC_CODE_RANGE_POLL 0xE1
#define FUNC_CODE_RANGE_RESP 0xE2
#define FUNC_CODE_RANGE_REPORT 0xE3
#define FUNC_CODE_GATEWAY_CMD 0xE4
#define FUNC_CODE_RESET 0xEF

#define MSG_LEN 64
#define BUF_LEN MSG_LEN
#define MSG_SN_IDX 2
#define MSG_DID_IDX 5
#define MSG_SID_IDX 7
#define MSG_FUNC_IDX 9
#define MSG_PAIR_IDX 10
#define MSG_FROM_UID_IDX 11
#define MSG_TO_UID_IDX 12
#define MSG_STATUS_IDX 13
#define MSG_DISTANCE_CM_IDX 14
#define RESP_MSG_POLL_RX_TS_IDX 16
#define RESP_MSG_RESP_TX_TS_IDX 20
#define MSG_RX_POWER_CDBM_IDX 24
#define MSG_FP_POWER_CDBM_IDX 26
#define MSG_REQUEST_ID_IDX 28
#define MSG_MEASURER_UID_IDX 29

#define REPORT_STATUS_OK 0
#define REPORT_STATUS_NO_RESPONSE 1
#define REPORT_STATUS_LOCAL_ERROR 2
#define REPORT_STATUS_INVALID_RANGE 3
#define REPORT_INVALID_CM 0xFFFF

#define PAIR_COUNT 15
#define PAIR_AB_INDEX 0
#define PAIR_BD_INDEX 2
#define PAIR_BF_INDEX 4
#define PAIR_CD_INDEX 5
#define PAIR_DE_INDEX 6
#define PAIR_EF_INDEX 7
#define PAIR_CE_INDEX 8
#define PAIR_DF_INDEX 10
#define PAIR_AC_INDEX 11
#define PAIR_AD_INDEX 12
#define PAIR_AE_INDEX 13

#define ENABLE_DETERMINISTIC_WALL_RECOVERY 1
#define COLLECTOR_PRINT_INTERVAL_MS 1000UL
#define MAX_DISTANCE_AGE_MS 3000UL

#define POLLING_PAIR_GAP_MS 10UL
#define POLLING_DIRECTION_RETRY_GAP_MS 5UL
#define POLLING_REPORT_WAIT_MS 120UL
#define GATEWAY_REPORT_WAIT_MS 450UL
#define POLLING_WORKER_RX_WINDOW_MS 100UL
#define POLLING_WORKER_SETTLE_MS 5UL
#define TX_DONE_TIMEOUT_MS 20UL

#define POLL_TX_TO_RESP_RX_DLY_UUS 300
#define POLL_RX_TO_RESP_TX_DLY_UUS 900
#define RESP_RX_TIMEOUT_UUS 3000
#define MAX_VALID_DISTANCE_M 200.0f

extern dwt_config_t config;

extern uint8_t tx_msg[], rx_msg[];
extern uint8_t frame_seq_nb;
extern uint8_t cycle_seq_nb;
extern uint32_t status_reg;
extern int ret;

extern float latest_distances[PAIR_COUNT];
extern float latest_rx_power_dbm[PAIR_COUNT];
extern float latest_fp_power_dbm[PAIR_COUNT];
extern bool latest_distance_valid[PAIR_COUNT];
extern bool latest_distance_recovered[PAIR_COUNT];
extern uint8_t latest_distance_seq[PAIR_COUNT];
extern unsigned long latest_distance_ms[PAIR_COUNT];

void start_uwb();
void polling_master();
void polling_node();

void initiator();
void responder();

#endif
