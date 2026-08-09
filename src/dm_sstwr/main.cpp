#include "dm_sstwr/uwb.h"
#include "dw3000.h"

void setup() {
    UART_init();
    spiBegin(UWB_IRQ, UWB_RST);
    spiSelect(UWB_SS);
    delay(2);
    start_uwb();
}

void loop() {
#ifdef POLLING_MASTER
    polling_master();
#else
    polling_node();
#endif
}
