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
		P |= 0x0777u;
		// P |= Bit(0u) | Bit(1u) | Bit(2u) | Bit(4u);
		info |= Bit(W_a);
	}
	if (buffer[1] == PIX_HIG) {
		P |= (0x0777u << 1u);
		// P |= Bit(1u) | Bit(2u) | Bit(3u);
		info |= Bit(W_b);
	}
	if (buffer[2] == PIX_HIG) {
		P |= (0x0777u << 4u);
		// P |= Bit(4u) | Bit(8u);
		info |= Bit(W_c);
	}
	if (buffer[3] == PIX_HIG) {
		info |= Bit(W_d);
	}

	if (coords.x == 0u) {
		// We can't connect left
		P &= 0xEEEEu;
		// P &= ~(Bit(0u) | Bit(4u) | Bit(5u) | Bit(8u) | Bit(9u));
	}
	if (coords.x + 1u >= dims.x) {
		P &= 0x3333u;
		// P &= ~(Bit(2u) | Bit(3u));
	} else if (coords.x + 2u >= dims.x) {
		P &= 0x7777u;
		// P &= ~(Bit(3u));
	}

	if (coords.y == 0u) {
		P &= 0xFFF0u;
		// P &= ~(Bit(0u) | Bit(1u) | Bit(2u) | Bit(3u) | Bit(6u) | Bit(7u));
	}
	if (coords.y + 1u >= dims.y) {
		P &= 0x00FFu;
		// P &= ~(Bit(8u) | Bit(9u));
	} else if (coords.y + 2u >= dims.y) {
		P &= 0x0FFFu;
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

		// Store data
		let white_father = uf_element(vec2u_offset(coords, father_offset));

		let ld = LabelData(
			white_father,
			info,
		);

		set_ld(coords, dims, ld);
    }
}