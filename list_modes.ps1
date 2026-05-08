Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DH7 {
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
    [DllImport("user32.dll")] public static extern int ChangeDisplaySettingsExA(string dev, ref DEVMODE dm, IntPtr hw, int fl, IntPtr lp);
    [DllImport("user32.dll")] public static extern int ChangeDisplaySettingsA(ref DEVMODE dm, int fl);
}
"@

# List all available modes on all displays
for ($d = 1; $d -le 4; $d++) {
    $dev = "\\.\DISPLAY$d"
    $dm = New-Object DH7+DEVMODE
    $dm.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf($dm)
    if ([DH7]::EnumDisplaySettingsA($dev, -1, [ref]$dm)) {
        Write-Host "$dev current: $($dm.dmPelsWidth)x$($dm.dmPelsHeight)@$($dm.dmDisplayFrequency)Hz"
        
        # Check for 1920x1200 modes
        $i = 0
        $modes1200 = @()
        while ([DH7]::EnumDisplaySettingsA($dev, $i, [ref]$dm)) {
            if ($dm.dmPelsWidth -eq 1920 -and $dm.dmPelsHeight -eq 1200) {
                $modes1200 += "$($dm.dmDisplayFrequency)Hz"
            }
            $i++
        }
        if ($modes1200.Count -gt 0) {
            $unique = ($modes1200 | Sort-Object -Unique) -join ", "
            Write-Host "  1920x1200 rates: $unique"
        }
    }
}
