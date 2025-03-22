$BaseUrl = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"
$DownloadDir = ".\downloads"

New-Item -ItemType Directory -Force -Path $DownloadDir

For ($i = 300; $i -le 492; $i++) {
    $FileName = "${i}_P.zip"
    $Url = "$BaseUrl/$FileName"
    $OutputPath = "$DownloadDir\$FileName"

    Write-Host "Downloading $FileName..."
    Invoke-WebRequest -Uri $Url -OutFile $OutputPath

    if (Test-Path $OutputPath) {
        Write-Host "$FileName downloaded successfully."
    } else {
        Write-Host "Failed to download $FileName."
    }
}

Write-Host "Download process completed."