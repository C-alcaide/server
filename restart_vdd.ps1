$ErrorActionPreference = "Continue"
"Starting VDD restart..." | Out-File "C:\VirtualDisplayDriver\restart_log.txt"
try {
    Disable-PnpDevice -InstanceId "ROOT\DISPLAY\0000" -Confirm:$false -ErrorAction Stop
    "Disabled OK" | Out-File "C:\VirtualDisplayDriver\restart_log.txt" -Append
} catch {
    "Disable failed: $_" | Out-File "C:\VirtualDisplayDriver\restart_log.txt" -Append
}
Start-Sleep -Seconds 3
try {
    Enable-PnpDevice -InstanceId "ROOT\DISPLAY\0000" -Confirm:$false -ErrorAction Stop
    "Enabled OK" | Out-File "C:\VirtualDisplayDriver\restart_log.txt" -Append
} catch {
    "Enable failed: $_" | Out-File "C:\VirtualDisplayDriver\restart_log.txt" -Append
}
Start-Sleep -Seconds 3
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Screen]::AllScreens | ForEach-Object {
    "$($_.DeviceName) $($_.Bounds.Width)x$($_.Bounds.Height) at ($($_.Bounds.X),$($_.Bounds.Y))"
} | Out-File "C:\VirtualDisplayDriver\restart_log.txt" -Append
"Done" | Out-File "C:\VirtualDisplayDriver\restart_log.txt" -Append
