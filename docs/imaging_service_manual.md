# Imaging Workstation Service Manual

## 1 Overview
This manual describes service procedures for the Aurora Imaging Workstation, covering the
Detector, the Gantry, the Console, and the Collimator. It is intended for the Field Engineer
performing installation, calibration, and fault recovery.

## 1.1 System Components
The Aurora Imaging Workstation consists of four field-replaceable units. The Detector captures
the signal. The Gantry positions the Detector and the Collimator. The Console runs the operator
software and the Smart Reading Protocol. The Collimator shapes the beam.

## 2 Calibration
The Detector requires calibration after any Detector replacement or Collimator adjustment.
Calibration aligns the Detector gain map and verifies the Collimator geometry. Run the
Calibration routine from the Console; the Console stores the Detector gain map on completion.

## 2.1 Detector Gain Calibration
Detector gain calibration measures the Detector response across the field. Start the Detector
gain calibration from the Console maintenance menu. The Detector must be warmed for thirty
minutes. On success the Console reports a Detector gain map checksum.

## 2.2 Collimator Alignment
Collimator alignment verifies that the Collimator blades track the Detector field. Use the
Collimator alignment fixture. If the Collimator blades drift, the Console raises error code
E-204.

## 3 Error Codes
The Console reports faults as error codes. Error code E-101 indicates a Detector communication
loss. Error code E-204 indicates a Collimator alignment fault. Error code E-330 indicates a
Gantry motion timeout. Each error code lists a Detector, Collimator, or Gantry corrective
action.

## 3.1 Error Code E-101 Detector Communication Loss
Error code E-101 means the Console lost communication with the Detector. Check the Detector
data cable, reseat the Detector interface board, and restart the Console. If error code E-101
persists, replace the Detector interface board.

## 3.2 Error Code E-204 Collimator Alignment Fault
Error code E-204 means the Collimator blades are out of alignment with the Detector field.
Run the Collimator alignment routine. If error code E-204 persists after alignment, replace the
Collimator blade motor.

## 3.3 Error Code E-330 Gantry Motion Timeout
Error code E-330 means the Gantry did not reach the commanded position in time. Check the Gantry
motion brake, clear any obstruction, and rerun the Gantry homing routine.

## 4 Smart Reading Protocol
The Smart Reading Protocol runs on the Console and arranges studies for review. The Smart
Reading Protocol learns operator layout preferences and applies them to new studies on the
Console display.
