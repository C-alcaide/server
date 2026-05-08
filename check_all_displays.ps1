Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DH6 {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct DEVMODE {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
        public short dmSpecVersion; public short dmDriverVersion; public short dmSize;
        public short dmDriverExtra; public int dmFields;
        public int dmPositionX; public int dmPositionY;
        public int dmDisplayOrientation; public int dmDisplayFixedOutput;
        public short dmColor; public short dmDuplex; public short dmYResolution;
        public short dmTTOption; public short dmCollate;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
        public short dmLogPixels; public int dmBitsPerPel;
        public int dmPelsWidth; public int dmPelsHeight;
        public int dmDisplayFlags; public int dmDisplayFrequency;
        public int dmICMMethod; public int dmICMIntent; public int dmMediaType;
        public int dmDitherType; public int dmReserved1; public int dmReserved2;
        public int dmPanningWidth; public int dmPanningHeight;
    }
    [DllImport("user32.dll")] public static extern bool EnumDisplaySettingsA(string dev, int mode, ref DEVMODE dm);
}
"@

for ($d = 1; $d -le 4; $d++) {
    $dev = "\\.\DISPLAY$d"
    $dm = New-Object DH6+DEVMODE
    $dm.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf($dm)
    if ([DH6]::EnumDisplaySettingsA($dev, -1, [ref]$dm)) {
        Write-Host "$dev current: $($dm.dmPelsWidth)x$($dm.dmPelsHeight)@$($dm.dmDisplayFrequency)Hz"
        $rates = @()
        $i = 0
        while ([DH6]::EnumDisplaySettingsA($dev, $i, [ref]$dm)) {
            $key = "$($dm.dmPelsWidth)x$($dm.dmPelsHeight)@$($dm.dmDisplayFrequency)"
            $rates += $key
            $i++
        }
        $unique = $rates | Sort-Object -Unique | Where-Object { $_ -match "1920x1200" -or $_ -match "@25$" -or $_ -match "@50$" }
        if ($unique) { Write-Host "  Relevant modes: $($unique -join ', ')" }
    }
}
