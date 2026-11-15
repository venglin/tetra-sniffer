# tetra-sniffer

A TETRA (ETSI EN 300 392-2) **band scanner and downlink demodulator/decoder** for the
**HackRF One**, written in Python with only `numpy` and `scipy`. The command-line tool
is called `tetrascan`.

It finds TETRA carriers in a band and identifies the cells (MCC, MNC, colour code,
location area, main carrier). It decodes the downlink signalling of one or more carriers
live or from recordings, and can forward every decoded block to Wireshark over GSMTAP.

```
$ tetrascan file examples/synthetic.cs8
auto-detected channels: 490.0500, 490.3000
     0.694 490.0500 TN2 FN18 MN01 BSCH   SYNC MCC 901 MNC 9999 CC 42 TN 2 FN 18 MN 1 continuous load 1
     0.737 490.0500 TN1 FN01 MN02 SCH/HD MAC-RESOURCE SSI 2345678 | BL-DATA CMCE D-TX-GRANTED call 102 granted talker 8765432
     0.765 490.0500 TN3 FN01 MN02 SCH/F  MAC-RESOURCE SSI 8765432 [frag start]
     0.794 490.0500 TN1 FN02 MN02 SCH/HD MAC-RESOURCE SSI 8765432 | BL-ADATA MM D-LOCATION-UPDATE-ACCEPT
     0.879 490.0500 TN3 FN03 MN02 SCH/F  MAC-END | BL-ADATA CMCE D-SDS-DATA from 2345678 bits 520 text messaging
     0.935 490.0500 TN3 FN04 MN02 SCH/F  MAC-RESOURCE SSI 8765432 | BL-DATA CMCE D-CONNECT call 102 granted prio 7
     0.964 490.3000 TN1 FN18 MN01 BNCH   SYSINFO main 490.3000 MHz (carrier 3612 band 4) UL 480.3000 LA 1001 HF 1 ...
     0.978 490.3000 TN2 FN18 MN01 BSCH   SYNC MCC 901 MNC 9999 CC 17 TN 2 FN 18 MN 1 continuous load 1
     0.992 490.0500 TN3 FN05 MN02 SCH/F  MAC-RESOURCE SSI broadcast | BL-UDATA MLE D-NWRK-BROADCAST time 2026-11-14 10:23:12 UTC (local +60 min) neighbours 2: [CA 1 carrier 3610 490.2500 MHz] ...
     1.020 490.0500 TN1 FN06 MN02 SCH/HD MAC-RESOURCE SSI 2345678 chan[replace carrier 3606 (490.1500 MHz) ts 2 UL+DL] | BL-ADATA CMCE D-SETUP call 103 speech group granted prio 3
     1.105 490.0500 TN3 FN07 MN02 SCH/F  MAC-RESOURCE SSI 2345678 | BL-ADATA CMCE D-SETUP call 1103 speech group granted prio 5 from 8765432
```

(The bundled recording is synthetic: two fictitious cells with MCC 901 near 490 MHz.)

## Features

- **Scanner.** A 20 Msps power survey covers 380–470 MHz in about one second. Every
  TETRA-like carrier (the right bandwidth, on the 6.25 kHz raster) is then decoded to
  confirm it and identify the cell. Secondary carriers, which send no synchronisation
  bursts, are identified with colour codes learned from the other carriers. The scanner
  also estimates the HackRF clock error from the cell's own SYSINFO.
- **Complete downlink physical layer.** π/4-DQPSK at 18 ksymbol/s: carrier offset
  acquisition (±10 kHz) and tracking, burst synchronisation on the training sequences,
  and normal and synchronisation burst types.
- **Coherent detection.** A max-log BCJR over the differential phase trellis with
  4th-power carrier phase tracking. It is about 2 dB more sensitive than the classic
  differential detector, which remains available (`--demod differential`).
- **Complete lower MAC.** Descrambling, de-interleaving, RCPC soft-decision Viterbi
  decoding and the CRC-16 for BSCH, BNCH, SCH/HD, STCH and SCH/F, plus RM(30,14)
  maximum-likelihood decoding of the AACH.
- **Secondary carriers.** Colour codes are shared between the carriers of one capture,
  and slot-aligned carriers also inherit the TDMA time. A known colour code can also be
  given with `--cell MCC:MNC:CC`.
- **Upper layers.**
  - MAC: MAC-SYNC / D-MLE-SYNC, SYSINFO / D-MLE-SYSINFO, ACCESS-DEFINE, ACCESS-ASSIGN,
    MAC-RESOURCE (addresses, channel allocations), MAC-FRAG / MAC-END reassembly.
  - LLC.
  - MLE: D-NWRK-BROADCAST with network time and neighbour cells.
  - MM and CMCE PDU types: call set-up, connect, transmission grants, releases, status,
    calling and transmitting party, SDS metadata.
- **Multi-carrier.** Decodes several carriers within 1.6 MHz from a single capture.
- **Wireshark ready.** Writes GSMTAP (`TETRA_I1`) live over UDP or into a PCAP file.
- **No GNU Radio, no compiled code.** libhackrf is driven directly through `ctypes`.

## Installation

```sh
git clone https://github.com/venglin/tetra-sniffer
cd tetra-sniffer
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

Requirements: Python ≥ 3.10, `numpy`, `scipy` and **libhackrf**:

- **Windows:** install [PothosSDR](https://downloads.myriadrf.org/builds/PothosSDR/).
  `hackrf.dll` is looked up in PothosSDR, radioconda, SDRangel and Universal Radio Hacker.
  Set `HACKRF_DLL` to the full path of the DLL to use another one.
- **Linux / macOS:** install libhackrf from your package manager (`libhackrf0`, or
  `hackrf` in Homebrew). The library is found via `ctypes.util.find_library`.

Development and on-air testing were done on Windows 11 with PothosSDR and a HackRF One.
The DSP and tests are platform independent. The HackRF path on Linux/macOS has not been
tested with hardware.

## Usage

```sh
# try it without hardware: decode the bundled synthetic recording
tetrascan file examples/synthetic.cs8

# find TETRA cells between 380 and 470 MHz (default band)
tetrascan scan

# decode two carriers of a cell live, correcting a +13 ppm HackRF clock error
tetrascan live -f 391.2125 -f 391.0125 --ppm 13

# a secondary carrier on its own: give the cell's colour code (as printed by scan)
tetrascan live -f 391.4125 --ppm 13 --cell 901:9999:4

# cell activity report (slot occupancy, traffic sessions, signalling load) every 10 minutes
tetrascan live -f 391.2125 --ppm 13 --report 600

# long-term monitoring: reports every 5 minutes into SQLite, then hourly / daily summaries
tetrascan live -f 391.2125 -f 391.4125 --ppm 13 -q --status 0 --log activity.db
tetrascan stats activity.db
tetrascan stats activity.db --by day

# ...and watch it in Wireshark (filter: gsmtap)
tetrascan live -f 391.2125 --gsmtap
tetrascan live -f 391.2125 --pcap cell.pcap

# record raw IQ (cs8, 2.304 Msps, 4.6 MB/s) with a .json sidecar, decode it later
tetrascan record -f 391.2125 -f 391.0125 -d 60 -o cell.cs8
tetrascan file cell.cs8

# synthetic benchmark: decoded blocks vs Es/N0 for both detectors
tetrascan selftest
```

`python -m tetrascan ...` works too, without installing the package.

### Scan output

This is an illustrative example with fictitious values:

```
$ tetrascan scan
survey 380.000-470.000 MHz done in 1.0 s, max clipping 0.00 %
21 carriers above 10.0 dB, 5 to verify
  freq MHz   SNR  width  status    MCC   MNC  CC     LA   MER  notes
  391.0125  25.3  22.0k  TETRA     901  9999   4   1504  17.9  cell main carrier 391.2125
  391.2125  21.9  22.0k  TETRA     901  9999   4   1504  15.4  main carrier
  391.4125   9.8  24.4k  TETRA     901  9999   4                6.1  secondary carrier, colour code from 391.2125 MHz
  394.4875   8.9  22.0k  TETRA?                           9.1  bursts locked, no colour code found
  402.4187  10.2  24.4k  no lock
HackRF clock error: +13.01 ppm (pass --ppm 13.0 to correct it)
```

Status values:

| status | meaning |
|---|---|
| `TETRA` | a BSCH passed its CRC, or blocks decoded with a colour code learned from another carrier. MCC / MNC / CC are shown, plus LA and the main carrier once a SYSINFO is decoded |
| `TETRA?` | the burst structure locked with clean π/4-DQPSK symbols, but no colour code was found within the dwell time |
| `no lock` | the carrier has a TETRA-like bandwidth but did not decode as TETRA |

Only the main carrier of a cell is guaranteed to carry the BSCH and BNCH. Carriers that
show `cell main carrier …` belong to the cell on that frequency.

### Common options

| option | meaning |
|---|---|
| `-f MHZ` | carrier frequency; repeat for several carriers within 1.6 MHz |
| `--ppm N` | HackRF clock error correction (`scan` prints the value to use) |
| `--lna N`, `--vga N` | HackRF gains (default 32 / 30 dB); lower them if the status line reports clipping |
| `--amp` | enable the HackRF +14 dB RF amplifier |
| `--demod coherent\|differential` | symbol detector (default `coherent`) |
| `--cell MCC:MNC:CC` | (`live`, `file`) known colour code for carriers without a BSCH; repeatable |
| `-v`, `-vv` | also print broadcast, null and access PDUs / also print AACH |
| `--json` | one JSON object per PDU (for scripting) |
| `--gsmtap [HOST:PORT]` | send decoded blocks to Wireshark over UDP (default 127.0.0.1:4729) |
| `--pcap FILE` | write decoded blocks as GSMTAP into a PCAP file |
| `--report [N]` | (`live`, `file`) cell activity report at the end, and every N s if given (see below) |
| `--log FILE` | (`live`, `file`) append the activity reports to SQLite (`.db`, `.sqlite`) or CSV (any other name); implies `--report 300` |
| `-q` | (`live`, `file`) do not print decoded PDUs (GSMTAP, reports and the log still work) |
| `--status N` | (`live`) status line every N s: lock, MER, level, carrier offset, CRC rate, clipping, USB drops |

`scan` also has `--start` / `--stop` (MHz), `--threshold` (dB), `--dwell` (s per verify
window), `--all` (verify every carrier, not only the TETRA-like ones) and `--no-verify`.

By default `live` and `file` print each cell's identity and SYSINFO once (and again if
they change), then every dedicated signalling PDU. Idle cells are therefore quiet. Use
`-v` to see everything.

### Activity report

`--report` summarises what a carrier is doing, also when all of its signalling is
encrypted. It needs no decryption: the AACH of every burst says, in clear, whether its
slot carries control, nothing, or a traffic channel (with a usage marker), and the
MAC-RESOURCE headers in front of encrypted signalling are clear as well. Only counts are
reported, never addresses or contents. An illustrative example with fictitious values:

```
[report] 391.4125 MHz  MCC 901 MNC 9999 CC 7 LA 2001  period 60.0 s
  TN1  DL common control 100% | UL control 100% | sync bursts 1%
  TN2  DL unallocated 100% | UL unallocated 100% | sync bursts 100%
  TN3  DL traffic 66%, unallocated 18%, assigned control 16%, 5 sessions | UL traffic 66%, ...
  TN4  DL unallocated 72%, traffic 15%, assigned control 13%, 4 sessions | UL unallocated 72%, ...
  traffic sessions: 9 ended, 1 ongoing, 8 usage markers; duration total 51.9 s, median 2.9 s, max 20.3 s
  signalling: 994 PDUs to 25 addresses (99% encrypted), 799 null, 157 fragmented, 29 random-access acks, 279 channel allocations (279 encrypted)
  clear L3: D-LOCATION-UPDATE-ACCEPT 4, D-AUTHENTICATION 2
```

- **TNn.** Share of each downlink usage (and of the matching uplink slot) seen on the
  AACH, excluding frame 18. `sync bursts` is the share of synchronisation bursts; an idle
  base station often fills unused slots with them.
- **Traffic sessions.** A session is one usage marker held on one slot. With transmission
  trunking the network assigns a channel per over, so a session is one press of the PTT;
  with message trunking it is a whole call.
- **Signalling.** Non-null MAC-RESOURCE PDUs, how many distinct addresses they carry and
  how many are encrypted; fragmented PDUs; random-access acknowledgements (answers to
  terminals transmitting on the uplink); channel allocations.
- **Clear L3.** Counts of unencrypted layer-3 PDU types.

With `--report N` each report covers the N seconds since the previous one, which suits
long-term monitoring. With `--json` every report is also printed as one JSON line.

### Long-term monitoring

`--log FILE` appends every report to a file, one row per carrier and period, with the
reception quality of that period (bursts, MER, level, CRC rate). Finished traffic
sessions go into a second table (`sessions`), or into `FILE.sessions.csv` for CSV logs,
with their start time, timeslot, usage marker and duration. As in the report, only
counts and durations are stored.

`tetrascan stats FILE [--by hour|day]` summarises a log. An illustrative example with
fictitious values:

```
hour              freq MHz  CC  cover   Erl  busy%  sess/h  avg s  PDU/min  addr  enc%  RA/min   MER  CRC%
2026-11-15 16:00  391.2125   4  60.0m  0.00    0.0     0.0      -      0.0     0     -     0.0  16.2   100
2026-11-15 16:00  391.4125   7  60.0m  0.67   44.0   775.4    3.1   1026.9    22   100    13.8   6.7   100
```

| column | meaning |
|---|---|
| `cover` | minutes of the hour (or day) that were monitored |
| `Erl` | traffic intensity: mean number of downlink timeslots carrying traffic |
| `busy%` | traffic share of the busiest timeslot |
| `sess/h`, `avg s` | traffic sessions per hour and their mean duration |
| `PDU/min`, `addr` | signalling PDUs per minute; most distinct addresses seen in one period |
| `enc%` | share of signalling PDUs that are encrypted |
| `RA/min` | random-access acknowledgements per minute |
| `MER`, `CRC%` | reception quality |

The SQLite file can also be queried directly, for example
`SELECT strftime('%H', t_start, 'unixepoch', 'localtime') AS h, count(*) FROM sessions GROUP BY h`.

## Performance

Results of `tetrascan selftest` on a desktop CPU. Each point is 4 s of a synthetic main
carrier with a random carrier offset (±8 kHz) and sample clock error (±30 ppm). The
table shows the share of signalling blocks decoded:

| Es/N0 | MER | coherent (default) | differential |
|---|---|---|---|
| 5 dB | 3.2 dB | 0 % | 0 % |
| 6 dB | 3.7 dB | 33 % | 14 % |
| 7 dB | 4.3 dB | 84 % | 49 % |
| 8 dB | 5.0 dB | 99.6 % | 88 % |
| ≥ 10 dB | ≥ 6.5 dB | 100 % | ≥ 99.6 % |

On a weak real cell (MER 4.7 dB) the coherent detector raised the CRC success rate from
74 % to 95 %. Speed for one carrier is about 7× real time with coherent detection and
11× with differential. Four carriers at 2.304 Msps decode at 1.6× real time on one core
(2.3× with differential detection). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for
the signal chain.

## Validation

- **Unit and end-to-end tests** (`pytest`, 37 tests): every coding stage, including the
  scrambler against a direct implementation of the recursion in clause 8.2.5.2, the burst
  layouts, both detectors, the PDU parsers, the streaming filters, colour-code sharing
  (including the scanner's second pass with a simulated HackRF) and a full synthetic
  receive chain with impairments.
- **Wireshark cross-check.** GSMTAP output was dissected by Wireshark's own TETRA
  dissector:
  - *Matching field values:* SYSINFO, MAC-RESOURCE, LLC, D-SETUP (with calling party),
    D-CONNECT, D-TX-GRANTED (with transmitting party), D-SDS-DATA and ACCESS-DEFINE.
  - *Known differences, in Wireshark's definitions:* it reads the ACCESS-DEFINE GSSI as
    25 bits instead of 24. It also expects a second 3-bit PDU type inside
    D-NWRK-BROADCAST, unlike its own CMCE definitions. tetrascan follows the single PDU
    type of the standard.
  - *Not checked:* Wireshark does not dissect GSMTAP SCH/HD blocks at all, and the
    neighbour cell information element is not defined in it, so that element has not been
    checked against an independent implementation.
- **On air:** idle and busy cells of live networks in the 380–430 MHz range.
  - Covered synchronisation and both normal burst types, and the logical channels BSCH,
    BNCH, AACH, SCH/HD, STCH and SCH/F, with 100 % CRC success at MER ≥ 15 dB.
  - Covered air-interface encrypted MAC-RESOURCE PDUs and fragmented signalling.
  - D-NWRK-BROADCAST and the optional CMCE elements have not been seen unencrypted on air
    yet; the networks tested use air-interface encryption.

## Not implemented

- **Voice.** Traffic channels (TCH) are detected and counted, but not decoded (no ACELP
  codec).
- **Decryption.** Encrypted TM-SDUs are reported as encrypted. This tool does not, and
  will not, attempt to decrypt anything.
- Uplink, Direct Mode Operation (DMO) and TEDS (QAM) carriers.
- Type-3 elements of L3 PDUs, and most MM PDU contents.
- SDS payloads. Only the sender, length and protocol identifier of short data messages
  are shown, never the content.

## Legal note

This is a receive-only tool for studying the TETRA air interface. In many countries
it is illegal to receive, decode or disclose radio communications that are not intended
for you, even unencrypted signalling. Check the law that applies to you, and only use
`tetrascan` where you are permitted to.

## References

- ETSI EN 300 392-2: *TETRA; Voice plus Data (V+D); Part 2: Air Interface (AI)*.
  Physical layer (clause 5), channel coding (clause 8), bursts (clause 9), MAC (clause 21),
  MLE (clause 18), CMCE (clause 14), MM (clause 16).
- [osmo-tetra](https://gitea.osmocom.org/tetra/osmo-tetra): the constants were
  cross-checked against it (no code was taken from it).
- Wireshark's TETRA and GSMTAP dissectors.

## License

MIT, see [LICENSE](LICENSE).
