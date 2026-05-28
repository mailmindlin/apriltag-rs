@group(0) @binding(1) var<storage, read_write> labels : Labels;
@group(0) @binding(2) var img_src : texture_2d<u32>;

struct ConnData {
	P: u32,
	// Bitmask representing two kinds of information
	// Bits 0, 1, 2, 3 are set if pixel a, b, c, d are foreground, respectively
	// Bits 4, 5, 6, 7 are set if block P, Q, R, S need to be merged to X in Merge phase
	info: u32,
}

fn process_chunk(coords: vec2u, buffer: array<u32, 4>) -> ConnData {
	let dims = textureDimensions(img_src);

	var P = 0u;
	var info = 0u;
	if buffer[0] == PIX_HIG {
		// P |= 0x0777u;
		P |= Bit(0u) | Bit(1u) | Bit(2u) | Bit(4u);
		info |= Bit(W_a);
	} else if buffer[0] == PIX_LOW {
		P |= Bit(5u) | Bit(6u);
		info |= Bit(B_a);
	}
	if (buffer[1] == PIX_HIG) {
		// P |= (0x0777u << 1u);
		P |= Bit(1u) | Bit(2u) | Bit(3u);
		info |= Bit(W_b);
	} else if buffer[1] == PIX_LOW {
		P |= Bit(7u);
		info |= Bit(B_b);
	}
	if (buffer[2] == PIX_HIG) {
		// P |= (0x0777u << 4u);
		P |= Bit(4u) | Bit(8u);
		info |= Bit(W_c);
	} else if buffer[2] == PIX_LOW {
		P |= Bit(9u);
		info |= Bit(B_c);
	}
	if (buffer[3] == PIX_HIG) {
		info |= Bit(W_d);
	} else if buffer[3] == PIX_LOW {
		info |= Bit(B_d);
	}

	if (coords.x == 0u) {
		// We can't connect left
		// P &= 0xEEEEu;
		P &= ~(Bit(0u) | Bit(4u) | Bit(5u) | Bit(8u) | Bit(9u));
	}
	if (coords.x + 1u >= dims.x) {
		// P &= 0x3333u;
		P &= ~(Bit(2u) | Bit(3u));
	} else if (coords.x + 2u >= dims.x) {
		// P &= 0x7777u;
		P &= ~(Bit(3u));
	}

	if (coords.y == 0u) {
		// P &= 0xFFF0u;
		P &= ~(Bit(0u) | Bit(1u) | Bit(2u) | Bit(3u) | Bit(6u) | Bit(7u));
	}
	if (coords.y + 1u >= dims.y) {
		// P &= 0x00FFu;
		P &= ~(Bit(8u) | Bit(9u));
	// } else if (coords.y + 2u >= dims.y) {
	// 	P &= 0x0FFFu;
	}
	return ConnData(P, info);
}

@compute
@workgroup_size(16, 16)
fn k04_bke_init(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let dims = textureDimensions(img_src);
	let coords = global_invocation_id.xy * vec2u(2u);

    if (all(coords < dims)) {
		let buffer = pixel4(coords);
		let chunk = process_chunk(coords, buffer);
		let P = chunk.P;
		var info = chunk.info;

        // P is now ready to be used to find neighbour blocks
        // P value avoids range errors

        var father_offset = vec2i(0);
		// var black_a_offset = vec2i(0, 0);
		// var black_b_offset = vec2i(1, 0);
		// var black_c_offset = vec2i(0, 1);
		// var black_d_offset = vec2i(1, 1);

        // P square (Ox W -> Px W)
        if HasBit(P, 0u) && pixel_high(coords, vec2(-1, -1)) {
			father_offset = vec2(-2, -2);
        }

        // Q square
        if (HasBit(P, 1u) && pixel_high(coords, vec2(0, -1))) || (HasBit(P, 2u) && pixel_high(coords, vec2(1, -1))) {
            if all(father_offset == vec2(0)) {
				father_offset = vec2(0, -2);
            } else {
				info |= Bit(BIT_Q);
            }
        }

		// Qc black
		// if HasBit(P, 6u) && pixel_low(coords, vec2(0, -1)) {
		// 	black_a_offset = vec2(0, -1);
		// 	if HasBit(info, B_b) {
		// 		black_b_offset = black_a_offset;
		// 	}
		// }

		// Qd black
		// if HasBit(P, 7u) && pixel_low(coords, vec2(1, -1)) {
		// 	if all(black_b_offset == vec2(1, 0)) {
		// 		black_b_offset = vec2(1, -1);
		// 		if all(black_a_offset == vec2(0, 0)) {
		// 			black_a_offset = black_b_offset;
		// 		}
		// 	} else {
		// 		info |= Bit(B_Qd);
		// 	}
		// }

		// a->Sb black
		// if HasBit(P, 5u) && pixel_low(coords, vec2(-1, 0)) {
		// 	if all(black_a_offset == vec2(0, 0)) {
		// 		black_a_offset = vec2(-1, 0);
		// 	} else {
		// 		info |= Bit(B_Sb);
		// 	}
		// }

		// Black c -> a
		// if HasBit(info, B_a) && HasBit(info, B_c) {
		// 	black_c_offset = black_a_offset;
		// }

        // R square
        if HasBit(P, 3u) && pixel_high(coords, vec2(2, -1)) {
            if all(father_offset == vec2(0)) {
				father_offset = vec2(2, -2);
            } else {
				info |= Bit(BIT_R);
            }
        }

        // S square
        if ((HasBit(P, 4u) && pixel_high(coords, vec2(-1, 0))) || (HasBit(P, 8u) && pixel_high(coords, vec2(-1, 1)))) {
            if all(father_offset == vec2(0)) {
                father_offset = vec2(-2, 0);
            } else {
				info |= Bit(BIT_S);
            }
        }

		// Black Sd
		// if HasBit(P, 9u) && pixel_low(coords, vec2(-1, 1)) {
		// 	if all(black_c_offset == vec2(0, 1)) {
		// 		black_c_offset = vec2(-1, 1);
		// 	} else {
		// 		info |= Bit(B_Sd);
		// 	}
		// }

		// Black d
		// if HasBit(info, B_d) {
		// 	if HasBit(info, B_b) {
		// 		black_d_offset = black_b_offset;
		// 	} else if HasBit(info, B_c) {
		// 		black_d_offset = black_c_offset;
		// 	}
		// }

		// Store data
		let white_father = uf_element(vec2u_offset(coords, father_offset));

		let ld = LabelData(
			white_father,
			info,
			// uf_element(vec2u_offset(coords, black_a_offset)),
			// uf_element(vec2u_offset(coords, black_b_offset)),
			// uf_element(vec2u_offset(coords, black_c_offset)),
			// uf_element(vec2u_offset(coords, black_d_offset)),
		);

		set_ld(coords, dims, ld);

		// let elem_a = uf_element(coords);
		// uf_parent_set(uf_element(coords), uf_element(coords));
		// uf_parent_set(uf_element(coords + vec2(1u, 0u)), uf_element(coords + vec2(1u, 0u)));
		// uf_parent_set(uf_element(coords + vec2(0u, 1u)), uf_element(coords + vec2(0u, 1u)));
		// uf_parent_set(uf_element(coords + vec2(1u, 1u)), uf_element(coords + vec2(1u, 1u)));
    }
}