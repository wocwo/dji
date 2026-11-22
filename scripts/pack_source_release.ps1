# 打包「源码 + 文档」供分享（排除向量库、大模型目录与超大数据文件）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Dist = Join-Path $Root "dist"
$StageName = "dji-rag-source-$Stamp"
$Stage = Join-Path $env:TEMP $StageName

if (Test-Path $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

# 根目录：代码与文档
Get-ChildItem -Path $Root -File | Where-Object {
    $_.Extension -in @(".py", ".md", ".txt") -or $_.Name -eq "requirements.txt"
} | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Stage $_.Name) }

# 小型 JSON 配置（排除已知巨型文件）
$skipJson = @("faiss_metadata_items.jsonl")
Get-ChildItem -Path $Root -File -Filter "*.json" | ForEach-Object {
    if ($skipJson -contains $_.Name) { return }
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Stage $_.Name)
}

if (Test-Path (Join-Path $Root "configs")) {
    Copy-Item -Path (Join-Path $Root "configs") -Destination (Join-Path $Stage "configs") -Recurse
}
if (Test-Path (Join-Path $Root "docs")) {
    Copy-Item -Path (Join-Path $Root "docs") -Destination (Join-Path $Stage "docs") -Recurse
}
if (Test-Path (Join-Path $Root "scripts")) {
    Copy-Item -Path (Join-Path $Root "scripts") -Destination (Join-Path $Stage "scripts") -Recurse
}

# 可选：样例块（若存在）
$sample = Join-Path $Root "sample_chunks.json"
if (Test-Path $sample) { Copy-Item -LiteralPath $sample -Destination (Join-Path $Stage "sample_chunks.json") }

$gitignore = Join-Path $Root ".gitignore"
if (Test-Path $gitignore) { Copy-Item -LiteralPath $gitignore -Destination (Join-Path $Stage ".gitignore") }

New-Item -ItemType Directory -Path $Dist -Force | Out-Null
$ZipPath = Join-Path $Dist "$StageName.zip"
if (Test-Path $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }

Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath -CompressionLevel Optimal
Remove-Item -LiteralPath $Stage -Recurse -Force

Write-Host "Created: $ZipPath"
