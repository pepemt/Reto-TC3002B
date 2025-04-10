import ftfy

# Texto con caracteres incorrectos
texto = "I wish I was prettier. I wish I didnâ€™t feel like a burden, I wish I wasnâ€™t so broken. I wish I was more charismatic and not weird around strangers and I wish I wasnâ€™t so nice. I wish I didnâ€™t feel like Iâ€™m a loser with a pathetic, boring life. I wish I wasnâ€™t so hard to make friends with and I just wish I was different."

# Arreglar el texto
texto_arreglado = ftfy.fix_text(texto)

print(texto_arreglado)  # Salida: Who’s