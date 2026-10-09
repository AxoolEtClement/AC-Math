.code

; Signature: extern "C" float Dot_Asm_x64(const void* lhs, const void* rhs);
; rcx = lhs pointer (this)
; rdx = rhs pointer
ALIGN 16
Dot_Asm_x64 PROC
    movaps  xmm0, [rcx]
    mulps   xmm0, [rdx]

    movaps  xmm1, xmm0          
    movhlps xmm1, xmm0          ; Move high half to low half
    addps   xmm0, xmm1

    movaps  xmm1, xmm0
    shufps  xmm1, xmm1, 1       ; Broadcast/shift lane 1 to lane 0
    addss   xmm0, xmm1

    ret
Dot_Asm_x64 ENDP

END