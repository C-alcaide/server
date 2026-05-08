Add-Type -AssemblyName System.Windows.Forms
Write-Host "`n=== Active Screens ==="
[System.Windows.Forms.Screen]::AllScreens | ForEach-Object {
    Write-Host "$($_.DeviceName) : $($_.Bounds.Width)x$($_.Bounds.Height) at ($($_.Bounds.X),$($_.Bounds.Y)) Primary=$($_.Primary)"
}

Write-Host "`n=== PnP Monitors ==="
Get-CimInstance -ClassName Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Monitor' } | ForEach-Object {
    Write-Host "$($_.Name) | $($_.DeviceID) | Status=$($_.Status)"
}

Write-Host "`n=== Display Adapters (WMI) ==="
Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, VideoProcessor, CurrentHorizontalResolution, CurrentVerticalResolution | Format-Table -AutoSize
