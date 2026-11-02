-- 从本地 Apple Music / Music 资料库按专辑导出内嵌封面。
-- 用法: osascript extract_apple_music_covers.applescript <输出目录> [manifest路径]
-- 输出: image_0001.jpg ... 以及 covers_manifest.tsv（序号、专辑、专辑艺人）
-- 旧目录不会被删除，只会写入指定的新目录。

on run argv
	if (count of argv) < 1 then error "用法: osascript extract_apple_music_covers.applescript <输出目录> [manifest路径]"
	set outDir to item 1 of argv
	if (count of argv) >= 2 then
		set manifestPath to item 2 of argv
	else
		set manifestPath to outDir & "/covers_manifest.tsv"
	end if
	do shell script "mkdir -p " & quoted form of outDir

	set sep to (ASCII character 31)
	set entries to {}
	set seen to {}
	set failed to {}
	set idx to 0
	set AppleScript's text item delimiters to sep

	tell application "Music"
		repeat with t in every track
			try
				set alb to album of t
				if alb is missing value then set alb to ""
				set aa to album artist of t
				if aa is missing value then set aa to ""
				set albumKey to (alb & sep & aa)

				set isNew to false
				ignoring case
					if seen does not contain albumKey then set isNew to true
				end ignoring

				if isNew and (count of artwork of t) > 0 then
					set idx to idx + 1
					set end of seen to albumKey
					set artworkData to data of artwork 1 of t
					if my writeData(artworkData, idx, outDir) then
						set end of entries to ((idx as text) & sep & alb & sep & aa)
					else
						set end of failed to {idx, alb, aa}
					end if
				end if
			end try
		end repeat

		-- 网络/缓存偶发失败，多轮延迟重试
		repeat 4 times
			set stillFailed to {}
			repeat with failedRef in failed
				set failedItem to contents of failedRef
				set failedIdx to item 1 of failedItem
				set failedAlbum to item 2 of failedItem
				set failedArtist to item 3 of failedItem
				try
					set matches to (every track whose album is failedAlbum and album artist is failedArtist)
					if (count of matches) is 0 then set matches to (every track whose album is failedAlbum)
					set t to item 1 of matches
					delay 0.4
					set artworkData to data of artwork 1 of t
					if my writeData(artworkData, failedIdx, outDir) then
						set end of entries to ((failedIdx as text) & sep & failedAlbum & sep & failedArtist)
					else
						set end of stillFailed to failedItem
					end if
				on error
					set end of stillFailed to failedItem
				end try
			end repeat
			set failed to stillFailed
			if (count of failed) is 0 then exit repeat
		end repeat
	end tell

	set mf to open for access (POSIX file manifestPath) with write permission
	set eof mf to 0
	repeat with lineText in entries
		write ((lineText as text) & linefeed) to mf as «class utf8»
	end repeat
	close access mf

	set ff to open for access (POSIX file (outDir & "/failed.tsv")) with write permission
	set eof ff to 0
	repeat with failedRef in failed
		set failedItem to contents of failedRef
		write (((item 1 of failedItem) as text) & sep & (item 2 of failedItem) & sep & (item 3 of failedItem) & linefeed) to ff as «class utf8»
	end repeat
	close access ff

	if (count of failed) is 0 then
		return "exported " & (count of entries) & " covers to " & outDir
	else
		return "exported " & (count of entries) & " covers, " & (count of failed) & " failed (see failed.tsv) in " & outDir
	end if
end run

on writeData(artworkData, idx, outDir)
	try
		set num to text 2 thru -1 of ((10000 + (idx as integer)) as text)
		set fileName to "image_" & num & ".jpg"
		set f to open for access (POSIX file (outDir & "/" & fileName)) with write permission
		set eof f to 0
		write artworkData to f
		close access f
		return true
	on error
		try
			close access f
		end try
		return false
	end try
end writeData
